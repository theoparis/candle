//! Strata-style MoE expert caching and disk streaming.
//!
//! Allows running large MoE models on memory-constrained devices by keeping only a bounded
//! working set of active/hot experts in device memory (VRAM / unified RAM) and streaming
//! uncached experts on demand from the GGUF file on disk.

use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{bail, Device, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Byte location and slice size of an expert's gate, up, and down projections in a file.
#[derive(Debug, Clone, Copy)]
pub struct ExpertLocation {
    pub gate_offset: u64,
    pub gate_bytes: usize,
    pub up_offset: u64,
    pub up_bytes: usize,
    pub down_offset: u64,
    pub down_bytes: usize,
}

/// Disk-backed expert store. Maps `(layer_idx, expert_idx)` to exact file slices.
#[derive(Debug)]
pub struct GgufExpertStore {
    mmap: memmap2::Mmap,
    /// locations[layer_idx][expert_idx]
    locations: HashMap<usize, Vec<ExpertLocation>>,
    pub dtype: GgmlDType,
    pub num_experts: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub bytes_per_proj: usize,

    // Diagnostics / telemetry
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub bytes_streamed: AtomicU64,
}

impl GgufExpertStore {
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        ct: &gguf_file::Content,
        _arch_prefix: &str,
        num_layers: usize,
        num_experts: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: The file is mapped read-only and is not modified concurrently.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let block_count = num_layers;
        let mut locations = HashMap::new();
        let mut detected_dtype = None;

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let gate_name = format!("{prefix}.ffn_gate_exps.weight");
            let up_name = format!("{prefix}.ffn_up_exps.weight");
            let down_name = format!("{prefix}.ffn_down_exps.weight");

            // Only MoE layers have expert tensors
            let (gate_info, up_info, down_info) = match (
                ct.tensor_infos.get(&gate_name),
                ct.tensor_infos.get(&up_name),
                ct.tensor_infos.get(&down_name),
            ) {
                (Some(g), Some(u), Some(d)) => (g, u, d),
                _ => continue,
            };

            let dtype = gate_info.ggml_dtype;
            detected_dtype = Some(dtype);

            let block_size = dtype.block_size();
            let type_size = dtype.type_size();

            let gate_elems_per_exp = intermediate_size * hidden_size;
            let up_elems_per_exp = intermediate_size * hidden_size;
            let down_elems_per_exp = hidden_size * intermediate_size;

            if !gate_elems_per_exp.is_multiple_of(block_size) {
                bail!(
                    "expert elements {gate_elems_per_exp} not divisible by block size {block_size}"
                );
            }

            let gate_bytes_per_exp = (gate_elems_per_exp / block_size) * type_size;
            let up_bytes_per_exp = (up_elems_per_exp / block_size) * type_size;
            let down_bytes_per_exp = (down_elems_per_exp / block_size) * type_size;

            let gate_base = ct.tensor_data_offset + gate_info.offset;
            let up_base = ct.tensor_data_offset + up_info.offset;
            let down_base = ct.tensor_data_offset + down_info.offset;

            let mut layer_locs = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                layer_locs.push(ExpertLocation {
                    gate_offset: gate_base + (e * gate_bytes_per_exp) as u64,
                    gate_bytes: gate_bytes_per_exp,
                    up_offset: up_base + (e * up_bytes_per_exp) as u64,
                    up_bytes: up_bytes_per_exp,
                    down_offset: down_base + (e * down_bytes_per_exp) as u64,
                    down_bytes: down_bytes_per_exp,
                });
            }
            locations.insert(layer_idx, layer_locs);
        }

        let dtype = match detected_dtype {
            Some(dt) => dt,
            None => bail!("no MoE expert layers detected in GGUF metadata"),
        };

        let bytes_per_proj =
            (intermediate_size * hidden_size / dtype.block_size()) * dtype.type_size();

        Ok(Self {
            mmap,
            locations,
            dtype,
            num_experts,
            intermediate_size,
            hidden_size,
            bytes_per_proj,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            bytes_streamed: AtomicU64::new(0),
        })
    }

    #[inline]
    pub fn read_proj(&self, offset: u64, bytes: usize) -> &[u8] {
        let start = offset as usize;
        let end = start + bytes;
        &self.mmap[start..end]
    }

    #[inline]
    pub fn get_location(&self, layer: usize, expert: usize) -> Option<ExpertLocation> {
        self.locations
            .get(&layer)
            .and_then(|exps| exps.get(expert).copied())
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
            self.bytes_streamed.load(Ordering::Relaxed),
        )
    }

    pub fn reset_stats(&self) {
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.bytes_streamed.store(0, Ordering::Relaxed);
    }
}

/// Per-layer slot-based LRU expert cache resident in device memory.
#[derive(Debug)]
pub struct LayerExpertCache {
    pub capacity: usize,
    pub gate_exps: QTensor,
    pub up_exps: QTensor,
    pub down_exps: QTensor,
    /// Which global expert ID occupies each slot index in 0..capacity.
    slots: Vec<Option<usize>>,
    /// Monotonic timestamp of last use for each slot.
    last_used: Vec<u64>,
    clock: u64,
}

impl LayerExpertCache {
    pub fn new(
        capacity: usize,
        intermediate_size: usize,
        hidden_size: usize,
        dtype: GgmlDType,
        device: &Device,
    ) -> Result<Self> {
        let gate_exps = QTensor::zeros((capacity, intermediate_size, hidden_size), dtype, device)?;
        let up_exps = QTensor::zeros((capacity, intermediate_size, hidden_size), dtype, device)?;
        let down_exps = QTensor::zeros((capacity, hidden_size, intermediate_size), dtype, device)?;

        Ok(Self {
            capacity,
            gate_exps,
            up_exps,
            down_exps,
            slots: vec![None; capacity],
            last_used: vec![0; capacity],
            clock: 0,
        })
    }

    /// Pre-populate slots with initial experts (e.g. experts 0..capacity) so the first tokens hit.
    pub fn warm_up(
        &mut self,
        source: &GgufExpertStore,
        layer_idx: usize,
        count: usize,
    ) -> Result<()> {
        let to_load = count.min(self.capacity).min(source.num_experts);
        for e in 0..to_load {
            let loc = match source.get_location(layer_idx, e) {
                Some(l) => l,
                None => continue,
            };
            let gate_bytes = source.read_proj(loc.gate_offset, loc.gate_bytes);
            let up_bytes = source.read_proj(loc.up_offset, loc.up_bytes);
            let down_bytes = source.read_proj(loc.down_offset, loc.down_bytes);

            self.gate_exps.write_bytes(e * loc.gate_bytes, gate_bytes)?;
            self.up_exps.write_bytes(e * loc.up_bytes, up_bytes)?;
            self.down_exps.write_bytes(e * loc.down_bytes, down_bytes)?;

            self.slots[e] = Some(e);
            self.last_used[e] = 0;
        }
        Ok(())
    }

    /// Ensures all requested `wanted_experts` are resident in the cache slots.
    /// Returns the local slot indices (each in `0..capacity`) corresponding to `wanted_experts`.
    pub fn ensure_resident(
        &mut self,
        source: &GgufExpertStore,
        layer_idx: usize,
        wanted_experts: &[usize],
    ) -> Result<Vec<u32>> {
        self.clock += 1;
        let now = self.clock;

        let mut local_slots = Vec::with_capacity(wanted_experts.len());
        let mut assigned_slots = Vec::with_capacity(wanted_experts.len());

        for &e in wanted_experts {
            // Check for a cache hit
            if let Some(s) = self.slots.iter().position(|&slot_e| slot_e == Some(e)) {
                source.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.last_used[s] = now;
                local_slots.push(s as u32);
                assigned_slots.push(s);
                continue;
            }

            // Cache miss: find an LRU slot that is NOT already assigned to another expert in this token
            source.cache_misses.fetch_add(1, Ordering::Relaxed);

            let mut best_slot = None;
            let mut oldest_time = u64::MAX;

            for s in 0..self.capacity {
                if assigned_slots.contains(&s) {
                    continue;
                }
                if self.slots[s].is_none() {
                    // Empty slot is highest priority
                    best_slot = Some(s);
                    break;
                }
                if self.last_used[s] < oldest_time {
                    oldest_time = self.last_used[s];
                    best_slot = Some(s);
                }
            }

            let s = match best_slot {
                Some(s) => s,
                None => bail!(
                    "capacity {} is too small for {} concurrent experts",
                    self.capacity,
                    wanted_experts.len()
                ),
            };

            // Stream missing expert from disk into slot s
            let loc = match source.get_location(layer_idx, e) {
                Some(l) => l,
                None => bail!("expert {e} out of bounds for layer {layer_idx}"),
            };

            let gate_data = source.read_proj(loc.gate_offset, loc.gate_bytes);
            let up_data = source.read_proj(loc.up_offset, loc.up_bytes);
            let down_data = source.read_proj(loc.down_offset, loc.down_bytes);

            self.gate_exps.write_bytes(s * loc.gate_bytes, gate_data)?;
            self.up_exps.write_bytes(s * loc.up_bytes, up_data)?;
            self.down_exps.write_bytes(s * loc.down_bytes, down_data)?;

            source.bytes_streamed.fetch_add(
                (loc.gate_bytes + loc.up_bytes + loc.down_bytes) as u64,
                Ordering::Relaxed,
            );

            self.slots[s] = Some(e);
            self.last_used[s] = now;
            local_slots.push(s as u32);
            assigned_slots.push(s);
        }

        Ok(local_slots)
    }
}
