//! Quantized LFM2.5 MoE (GGUF architecture `lfm2moe`).
//!
//! A hybrid stack: most blocks are short convolutions, a few (`head_count_kv > 0`) are
//! grouped-query attention. The first `leading_dense_block_count` FFNs are dense SwiGLU;
//! the rest are routed MoE with a sigmoid router, a selection-only bias (`exp_probs_b`),
//! and top-k weights taken from the unbiased scores.
//!
//! Attention uses a plain per-sequence KV cache (no paging yet). Conv layers keep a
//! fixed-size per-sequence state.

use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::{gguf_file, QMatMul, QTensor};
use candle::{bail, DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Linear, Module};
use std::collections::HashMap;
use std::sync::Arc;

fn get_qtensor<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    names: &[String],
) -> Result<QTensor> {
    for name in names {
        if let Ok(t) = ct.tensor(reader, name, device) {
            return Ok(t);
        }
    }
    bail!("cannot find tensor info for {}", names.join(" | "))
}

fn get_dequantized<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    names: &[String],
) -> Result<Tensor> {
    get_qtensor(ct, reader, device, names)?.dequantize(device)
}

#[derive(Debug, Clone)]
struct Mlp {
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.w1.forward(xs)?;
        let w3 = self.w3.forward(xs)?;
        self.w2.forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
enum MoeBackend {
    Static {
        gate_experts: Arc<QTensor>,
        up_experts: Arc<QTensor>,
        down_experts: Arc<QTensor>,
    },
    Streaming {
        cache: Arc<std::sync::Mutex<crate::expert_cache::LayerExpertCache>>,
        store: Arc<crate::expert_cache::GgufExpertStore>,
        layer_idx: usize,
    },
}

#[derive(Debug, Clone)]
struct MoeMlp {
    /// Router, `[num_experts, hidden]`, kept in F32.
    gate: Linear,
    /// Selection-only bias, `[1, num_experts]`. Absent in some exports.
    expert_bias: Option<Tensor>,
    backend: MoeBackend,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
}

impl MoeMlp {
    /// `xs` is `[batch, seq, hidden]`; returns F32 `[batch, seq, hidden]`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let n_tok = b_sz * seq_len;
        let xs = xs
            .reshape((n_tok, hidden))?
            .to_dtype(DType::F32)?
            .contiguous()?;

        // Scores are sigmoid(router logits). The bias only chooses the experts;
        // the weights are the unbiased scores of the chosen experts.
        let scores = candle_nn::ops::sigmoid(&self.gate.forward(&xs)?)?;
        let selection = match &self.expert_bias {
            Some(bias) => scores.broadcast_add(bias)?,
            None => scores.clone(),
        };
        let topk_ids = selection
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let mut topk_w = scores.gather(&topk_ids, D::Minus1)?;
        if self.norm_topk_prob {
            topk_w = topk_w.broadcast_div(&topk_w.sum_keepdim(D::Minus1)?)?;
        }
        if self.routed_scaling_factor != 1.0 {
            topk_w = (topk_w * self.routed_scaling_factor)?;
        }

        let mut _cache_guard = None;
        let (gate_experts, up_experts, down_experts, mapped_topk_ids): (
            &QTensor,
            &QTensor,
            &QTensor,
            Tensor,
        ) = match &self.backend {
            MoeBackend::Static {
                gate_experts,
                up_experts,
                down_experts,
            } => (
                gate_experts.as_ref(),
                up_experts.as_ref(),
                down_experts.as_ref(),
                topk_ids,
            ),
            MoeBackend::Streaming {
                cache,
                store,
                layer_idx,
            } => {
                let mut guard = cache
                    .lock()
                    .map_err(|e| candle::Error::msg(e.to_string()))?;
                let topk_vec = topk_ids.flatten_all()?.to_vec1::<u32>()?;
                let wanted: Vec<usize> = topk_vec.into_iter().map(|id| id as usize).collect();
                let local_slots = guard.ensure_resident(store, *layer_idx, &wanted)?;
                let dev = xs.device();
                let local_ids =
                    Tensor::new(local_slots.as_slice(), dev)?.reshape(topk_ids.shape())?;
                _cache_guard = Some(guard);
                let g = _cache_guard.as_ref().unwrap();
                (&g.gate_exps, &g.up_exps, &g.down_exps, local_ids)
            }
        };

        let ys = if xs.device().is_metal() {
            // Metal indexed kernels: F32 contiguous input, U32 ids, weights applied after.
            let x3 = xs.reshape((n_tok, 1, hidden))?.contiguous()?;
            let gate = gate_experts.indexed_moe_forward(&x3, &mapped_topk_ids)?;
            let up = up_experts.indexed_moe_forward(&x3, &mapped_topk_ids)?;
            let down_in = (up * candle_nn::ops::silu(&gate)?)?;
            let down = down_experts.indexed_moe_forward(&down_in, &mapped_topk_ids)?;
            down.broadcast_mul(&topk_w.unsqueeze(D::Minus1)?)?
        } else {
            // Group the (token, slot) pairs by expert for the CPU grouped GEMM.
            let is_prefill = n_tok > 1;
            let (expert_ids, sorted_token_ids) =
                mapped_topk_ids.flatten_all()?.sort_last_dim(true)?;
            let k = self.num_experts_per_tok;
            let gate = candle_nn::moe::moe_gemm_gguf(
                &xs,
                &gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                k,
                is_prefill,
                DType::F32,
            )?;
            let up = candle_nn::moe::moe_gemm_gguf(
                &xs,
                &up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                k,
                is_prefill,
                DType::F32,
            )?;
            let down_in = (up * candle_nn::ops::silu(&gate)?)?;
            candle_nn::moe::moe_gemm_gguf(
                &down_in,
                &down_experts,
                &Some(topk_w.clone()),
                &sorted_token_ids,
                &expert_ids,
                k,
                is_prefill,
                DType::F32,
            )?
        };
        ys.reshape((n_tok, (), hidden))?
            .sum(D::Minus2)?
            .reshape((b_sz, seq_len, hidden))
    }
}

#[derive(Debug, Clone)]
enum FeedForward {
    Dense(Mlp),
    Moe(MoeMlp),
}

impl FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(xs),
            Self::Moe(m) => m.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

#[derive(Debug, Clone)]
struct ShortConvLayer {
    in_proj: QMatMul,
    out_proj: QMatMul,
    /// `[hidden, l_cache]`
    conv: Tensor,
    l_cache: usize,
    cache: Option<Tensor>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum LayerKind {
    Attention(AttentionLayer),
    ShortConv(ShortConvLayer),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn: FeedForward,
    kind: LayerKind,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

impl AttentionLayer {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, _n, seq_len, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = xs.dims3()?;

        let q = self.wq.forward(xs)?;
        let k = self.wk.forward(xs)?;
        let v = self.wv.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    (
                        Tensor::cat(&[k_cache, &k], 2)?,
                        Tensor::cat(&[v_cache, &v], 2)?,
                    )
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.wo.forward(&y)
    }
}

impl ShortConvLayer {
    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let bcx = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, hidden)?;
        let c = bcx.narrow(1, hidden, hidden)?;
        let x = bcx.narrow(1, 2 * hidden, hidden)?;
        let bx = (b * &x)?.contiguous()?;

        let conv_weight = self.conv.contiguous()?;

        let conv_out = if seq_len == 1 {
            let mut state = match &self.cache {
                Some(cache) => cache.clone(),
                None => Tensor::zeros((b_sz, hidden, self.l_cache), bx.dtype(), bx.device())?,
            };
            state = if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                Tensor::cat(&[tail, bx.clone()], 2)?
            } else {
                bx.clone()
            };
            self.cache = Some(state.clone());

            (state * &conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
        } else {
            let conv = Conv1d::new(
                conv_weight
                    .reshape((hidden, 1, self.l_cache))?
                    .contiguous()?,
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: hidden,
                    ..Default::default()
                },
            );
            let out = conv.forward(&bx.contiguous()?)?.narrow(2, 0, seq_len)?;

            if self.l_cache > 0 {
                let cur_len = bx.dim(2)?;
                let start = cur_len.saturating_sub(self.l_cache);
                let mut cache_src = bx.narrow(2, start, cur_len - start)?;
                if cache_src.dim(2)? < self.l_cache {
                    let pad = self.l_cache - cache_src.dim(2)?;
                    let zeros =
                        Tensor::zeros((b_sz, hidden, pad), cache_src.dtype(), cache_src.device())?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                self.cache = Some(cache_src);
            }
            out
        };

        let conv_out = (c * &conv_out)?;
        let conv_out = conv_out.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<(usize, usize), Tensor>,
    pub expert_store: Option<Arc<crate::expert_cache::GgufExpertStore>>,
}

fn value_to_usize(v: &gguf_file::Value) -> Result<usize> {
    use gguf_file::Value::*;
    match v {
        U8(x) => Ok(*x as usize),
        I8(x) => Ok(*x as usize),
        U16(x) => Ok(*x as usize),
        I16(x) => Ok(*x as usize),
        U32(x) => Ok(*x as usize),
        I32(x) => Ok(*x as usize),
        U64(x) => Ok(*x as usize),
        I64(x) => Ok(*x as usize),
        F32(x) => Ok(*x as usize),
        F64(x) => Ok(*x as usize),
        Bool(x) => Ok(usize::from(*x)),
        String(_) => bail!("unexpected string metadata"),
        Array(_) => bail!("array should be handled separately"),
    }
}

fn read_usize_list(v: &gguf_file::Value, len: usize) -> Result<Vec<usize>> {
    match v {
        gguf_file::Value::Array(arr) => {
            let out = arr.iter().map(value_to_usize).collect::<Result<Vec<_>>>()?;
            if out.len() == len {
                Ok(out)
            } else if out.len() == 1 {
                Ok(vec![out[0]; len])
            } else {
                bail!(
                    "unexpected array length in metadata, expected {len} got {}",
                    out.len()
                )
            }
        }
        _ => Ok(vec![value_to_usize(v)?; len]),
    }
}

fn metadata<'a>(ct: &'a gguf_file::Content, key: &str) -> Result<&'a gguf_file::Value> {
    match ct.metadata.get(key) {
        Some(v) => Ok(v),
        None => bail!("cannot find {key} in metadata"),
    }
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_cache_capacity(ct, reader, device, None, None)
    }

    pub fn from_gguf_with_cache_capacity<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        gguf_path: Option<&std::path::Path>,
        resident_expert_capacity: Option<usize>,
    ) -> Result<Self> {
        let arch = match metadata(&ct, "general.architecture")? {
            gguf_file::Value::String(s) => s.clone(),
            _ => bail!("general.architecture is not a string"),
        };
        if arch != "lfm2moe" {
            bail!("expected a lfm2moe GGUF, got architecture {arch}");
        }
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let head_count = metadata(&ct, &key("attention.head_count"))?.to_u32()? as usize;
        let embedding_length = metadata(&ct, &key("embedding_length"))?.to_u32()? as usize;
        let context_length = metadata(&ct, &key("context_length"))?.to_u32()? as usize;
        let block_count = metadata(&ct, &key("block_count"))?.to_u32()? as usize;
        let rms_norm_eps =
            metadata(&ct, &key("attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;
        let rope_freq_base = metadata(&ct, &key("rope.freq_base"))?.to_f32()?;
        let l_cache = metadata(&ct, &key("shortconv.l_cache"))?.to_u32()? as usize;
        let head_count_kv =
            read_usize_list(metadata(&ct, &key("attention.head_count_kv"))?, block_count)?;

        let num_experts = metadata(&ct, &key("expert_count"))?.to_u32()? as usize;
        let num_experts_per_tok = metadata(&ct, &key("expert_used_count"))?.to_u32()? as usize;
        let intermediate_size =
            metadata(&ct, &key("expert_feed_forward_length"))?.to_u32()? as usize;
        let leading_dense = ct
            .metadata
            .get(&key("leading_dense_block_count"))
            .map(|v| v.to_u32())
            .transpose()?
            .unwrap_or(0) as usize;
        // llama.cpp enum: 2 = sigmoid. This is the only gating function implemented here.
        let gating = ct
            .metadata
            .get(&key("expert_gating_func"))
            .map(|v| v.to_u32())
            .transpose()?
            .unwrap_or(2);
        if gating != 2 {
            bail!("unsupported expert_gating_func {gating}; only sigmoid (2) is implemented");
        }
        // Not stored in the GGUF; these match the reference model config.
        let norm_topk_prob = match ct.metadata.get(&key("expert_weights_norm")) {
            Some(v) => v.to_bool()?,
            None => true,
        };
        let routed_scaling_factor = match ct.metadata.get(&key("expert_weights_scale")) {
            Some(v) => v.to_f32()? as f64,
            None => 1.0,
        };

        let expert_store = match (resident_expert_capacity, gguf_path) {
            (Some(cap), Some(path)) if cap < num_experts => {
                let store = crate::expert_cache::GgufExpertStore::from_file(
                    path,
                    &ct,
                    &arch,
                    block_count,
                    num_experts,
                    intermediate_size,
                    embedding_length,
                )?;
                Some(Arc::new(store))
            }
            _ => None,
        };

        let head_dim = embedding_length / head_count;
        let (cos, sin) = precompute_freqs_cis(head_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // The embedding table is tied to the output head. Keep the table in F16 to save
        // memory; the output head stays quantized.
        let tok_embd_names = [
            "token_embd.weight",
            "tok_embeddings.weight",
            "model.embed_tokens.weight",
        ]
        .map(String::from);
        let tok_embd_q = get_qtensor(&ct, reader, device, &tok_embd_names)?;
        let tok_embeddings = tok_embd_q.dequantize(device)?.to_dtype(DType::F16)?;

        let norm = RmsNorm::from_qtensor(
            get_qtensor(
                &ct,
                reader,
                device,
                &[
                    "output_norm.weight",
                    "embedding_norm.weight",
                    "token_embd_norm.weight",
                ]
                .map(String::from),
            )?,
            rms_norm_eps,
        )?;

        let output_q = match get_qtensor(
            &ct,
            reader,
            device,
            &["output.weight", "lm_head.weight"].map(String::from),
        ) {
            Ok(q) => q,
            Err(_) => get_qtensor(&ct, reader, device, &tok_embd_names)?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let is_attention = head_count_kv[layer_idx] > 0;

            let operator_norm =
                get_qtensor(&ct, reader, device, &[format!("{prefix}.attn_norm.weight")])?;
            let ffn_norm =
                get_qtensor(&ct, reader, device, &[format!("{prefix}.ffn_norm.weight")])?;

            let ffn = if layer_idx < leading_dense {
                FeedForward::Dense(Mlp {
                    w1: QMatMul::from_qtensor(get_qtensor(
                        &ct,
                        reader,
                        device,
                        &[format!("{prefix}.ffn_gate.weight")],
                    )?)?,
                    w2: QMatMul::from_qtensor(get_qtensor(
                        &ct,
                        reader,
                        device,
                        &[format!("{prefix}.ffn_down.weight")],
                    )?)?,
                    w3: QMatMul::from_qtensor(get_qtensor(
                        &ct,
                        reader,
                        device,
                        &[format!("{prefix}.ffn_up.weight")],
                    )?)?,
                })
            } else if let Some(store) = &expert_store {
                let gate_w = get_dequantized(
                    &ct,
                    reader,
                    device,
                    &[format!("{prefix}.ffn_gate_inp.weight")],
                )?
                .to_dtype(DType::F32)?;
                let bias_name = format!("{prefix}.exp_probs_b.bias");
                let expert_bias = if ct.tensor_infos.contains_key(&bias_name) {
                    Some(
                        get_dequantized(&ct, reader, device, &[bias_name])?
                            .to_dtype(DType::F32)?
                            .reshape((1, num_experts))?,
                    )
                } else {
                    None
                };
                let cap = resident_expert_capacity.unwrap().max(num_experts_per_tok);
                let mut cache = crate::expert_cache::LayerExpertCache::new(
                    cap,
                    intermediate_size,
                    embedding_length,
                    store.dtype,
                    device,
                )?;
                // Warm up first cap experts so initial prompt processing has fast hits
                cache.warm_up(store, layer_idx, cap)?;
                FeedForward::Moe(MoeMlp {
                    gate: Linear::new(gate_w, None),
                    expert_bias,
                    backend: MoeBackend::Streaming {
                        cache: Arc::new(std::sync::Mutex::new(cache)),
                        store: store.clone(),
                        layer_idx,
                    },
                    num_experts_per_tok,
                    norm_topk_prob,
                    routed_scaling_factor,
                })
            } else {
                let gate_w = get_dequantized(
                    &ct,
                    reader,
                    device,
                    &[format!("{prefix}.ffn_gate_inp.weight")],
                )?
                .to_dtype(DType::F32)?;
                let bias_name = format!("{prefix}.exp_probs_b.bias");
                let expert_bias = if ct.tensor_infos.contains_key(&bias_name) {
                    Some(
                        get_dequantized(&ct, reader, device, &[bias_name])?
                            .to_dtype(DType::F32)?
                            .reshape((1, num_experts))?,
                    )
                } else {
                    None
                };
                let mut experts = |suffix: &str| -> Result<Arc<QTensor>> {
                    Ok(Arc::new(get_qtensor(
                        &ct,
                        reader,
                        device,
                        &[format!("{prefix}.{suffix}.weight")],
                    )?))
                };
                FeedForward::Moe(MoeMlp {
                    gate: Linear::new(gate_w, None),
                    expert_bias,
                    backend: MoeBackend::Static {
                        gate_experts: experts("ffn_gate_exps")?,
                        up_experts: experts("ffn_up_exps")?,
                        down_experts: experts("ffn_down_exps")?,
                    },
                    num_experts_per_tok,
                    norm_topk_prob,
                    routed_scaling_factor,
                })
            };

            let kind = if is_attention {
                let wq = get_qtensor(&ct, reader, device, &[format!("{prefix}.attn_q.weight")])?;
                let wk = get_qtensor(&ct, reader, device, &[format!("{prefix}.attn_k.weight")])?;
                let wv = get_qtensor(&ct, reader, device, &[format!("{prefix}.attn_v.weight")])?;
                let wo = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[format!("{prefix}.attn_output.weight")],
                )?;
                let q_norm = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[format!("{prefix}.attn_q_norm.weight")],
                )?;
                let k_norm = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[format!("{prefix}.attn_k_norm.weight")],
                )?;
                LayerKind::Attention(AttentionLayer {
                    wq: QMatMul::from_qtensor(wq)?,
                    wk: QMatMul::from_qtensor(wk)?,
                    wv: QMatMul::from_qtensor(wv)?,
                    wo: QMatMul::from_qtensor(wo)?,
                    q_norm: RmsNorm::from_qtensor(q_norm, rms_norm_eps)?,
                    k_norm: RmsNorm::from_qtensor(k_norm, rms_norm_eps)?,
                    n_head: head_count,
                    n_kv_head: head_count_kv[layer_idx],
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    kv_cache: None,
                })
            } else {
                let in_proj = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[format!("{prefix}.shortconv.in_proj.weight")],
                )?;
                let out_proj = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[format!("{prefix}.shortconv.out_proj.weight")],
                )?;
                let conv = get_dequantized(
                    &ct,
                    reader,
                    device,
                    &[format!("{prefix}.shortconv.conv.weight")],
                )?;
                LayerKind::ShortConv(ShortConvLayer {
                    in_proj: QMatMul::from_qtensor(in_proj)?,
                    out_proj: QMatMul::from_qtensor(out_proj)?,
                    conv,
                    l_cache,
                    cache: None,
                })
            };

            layers.push(LayerWeights {
                operator_norm: RmsNorm::from_qtensor(operator_norm, rms_norm_eps)?,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                ffn,
                kind,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output_q)?,
            masks: HashMap::new(),
            expert_store,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };

        let mut hidden = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?;
        for layer in self.layers.iter_mut() {
            let residual = hidden.clone();
            let normed = layer.operator_norm.forward(&hidden)?;
            hidden = match &mut layer.kind {
                LayerKind::Attention(attn) => attn.forward(&normed, mask.as_ref(), index_pos)?,
                LayerKind::ShortConv(conv) => conv.forward(&normed)?,
            };
            hidden = (hidden + residual)?;

            let residual = hidden.clone();
            let ff = layer.ffn_norm.forward(&hidden)?;
            hidden = (layer.ffn.forward(&ff)? + residual)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden.i((.., seq_len - 1, ..))?;
        self.output.forward(&hidden)
    }
}
