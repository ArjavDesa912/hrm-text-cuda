use cudarc::driver::*;
use half::bf16;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum HrmError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("CUDA BLAS error: {0}")]
    Blas(#[from] cudarc::cublas::result::CublasError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type CudaResult<T> = Result<T, HrmError>;

/// Hyperparameters for HRM-Text-1B (size B).
pub struct HrmConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub n_layers: usize, // per module (e.g. 16 for H_module + 16 for L_module)
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub init_std: f32,
    pub embedding_scale: f32,
}

impl Default for HrmConfig {
    fn default() -> Self {
        let hidden = 1536usize;
        let init_std = 1.0 / (hidden as f32).sqrt(); // lecun_normal
        Self {
            hidden_size: hidden,
            num_heads: 12,
            head_dim: 128,
            intermediate_size: 4096,
            h_cycles: 2,
            l_cycles: 3,
            n_layers: 16,
            max_seq_len: 4096,
            vocab_size: 65536,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
            init_std,
            embedding_scale: 39.191835884530846,
        }
    }
}

/// Weights for a single Transformer layer.
pub struct LayerWeights {
    pub gqkv_proj: CudaSlice<bf16>, // [4*hidden, hidden]
    pub o_proj: CudaSlice<bf16>,    // [hidden, hidden]
    pub gate_proj: CudaSlice<bf16>, // [intermediate, hidden]  (first half of gate_up_proj)
    pub up_proj: CudaSlice<bf16>,   // [intermediate, hidden]  (second half of gate_up_proj)
    pub down_proj: CudaSlice<bf16>, // [hidden, intermediate]
}

/// Weights for one Transformer module (H or L).
pub struct ModuleWeights {
    pub layers: Vec<LayerWeights>,
}

/// Full model weights.
pub struct ModelWeights {
    pub h_module: ModuleWeights,
    pub l_module: ModuleWeights,
    pub embed_tokens: CudaSlice<bf16>,
    pub lm_head: CudaSlice<bf16>,
    pub z_l_init: CudaSlice<bf16>,
}

/// Pre-allocated scratch buffers to avoid per-call allocations.
pub struct ScratchBuffers {
    // Attention intermediates
    pub norm_buf: CudaSlice<bf16>,
    pub gqkv_buf: CudaSlice<bf16>,
    pub gate_buf: CudaSlice<bf16>,
    pub q_buf: CudaSlice<bf16>,
    pub k_buf: CudaSlice<bf16>,
    pub v_buf: CudaSlice<bf16>,
    pub attn_out_buf: CudaSlice<bf16>,
    pub o_out_buf: CudaSlice<bf16>,
    // FFN intermediates
    pub norm2_buf: CudaSlice<bf16>,
    pub ffn_gate_buf: CudaSlice<bf16>,
    pub ffn_up_buf: CudaSlice<bf16>,
    pub ffn_out_buf: CudaSlice<bf16>,
    pub down_buf: CudaSlice<bf16>,
    pub residual_buf: CudaSlice<bf16>,
    // Forward pass buffers
    pub tmp1: CudaSlice<bf16>,
    pub tmp2: CudaSlice<bf16>,
    pub mask_buf: CudaSlice<bf16>,
    // Logits buffer
    pub logits_buf: CudaSlice<bf16>,
    /// The (rows, seq_len) this scratch was allocated for
    alloc_rows: usize,
    alloc_seq: usize,
}

/// Per-slot KV cache for autoregressive decode.
pub struct KvCache {
    /// Per-slot K cache: [batch, heads, max_seq, head_dim] per slot
    pub k_cache: Vec<CudaSlice<bf16>>,
    /// Per-slot V cache: [batch, heads, max_seq, head_dim] per slot
    pub v_cache: Vec<CudaSlice<bf16>>,
    /// Current cached sequence length
    pub cached_len: usize,
    /// Max sequence length the cache was allocated for
    pub max_len: usize,
}

impl KvCache {
    fn new(
        dev: &Arc<CudaDevice>,
        config: &HrmConfig,
        batch_size: usize,
        max_len: usize,
    ) -> CudaResult<Self> {
        let total_attn_ops = config.h_cycles * (config.l_cycles + 1) * config.n_layers;
        let slot_size = batch_size * config.num_heads * max_len * config.head_dim;
        let mut k_cache = Vec::with_capacity(total_attn_ops);
        let mut v_cache = Vec::with_capacity(total_attn_ops);
        for _ in 0..total_attn_ops {
            let mut kb = unsafe { dev.alloc::<bf16>(slot_size)? };
            dev.memset_zeros(&mut kb)?;
            k_cache.push(kb);
            let mut vb = unsafe { dev.alloc::<bf16>(slot_size)? };
            dev.memset_zeros(&mut vb)?;
            v_cache.push(vb);
        }
        Ok(KvCache {
            k_cache,
            v_cache,
            cached_len: 0,
            max_len,
        })
    }
}

/// Owns device buffers and kernels for the full forward pass.
pub struct HrmForwardPass {
    pub dev: Arc<CudaDevice>,
    pub config: HrmConfig,
    pub weights: Option<ModelWeights>,
    pub blas: cudarc::cublas::CudaBlas,
    pub scratch: Option<ScratchBuffers>,
    pub kv_cache: Option<KvCache>,
    // Persistent buffers to avoid per-forward-pass allocations
    pub z_h_buf: Option<CudaSlice<bf16>>,
    pub z_l_buf: Option<CudaSlice<bf16>>,
    pub normed_buf: Option<CudaSlice<bf16>>,
    pub decode_mask_buf: Option<CudaSlice<bf16>>,
    // Kernel functions
    pub additive_inject: CudaFunction,
    pub rms_norm: CudaFunction,
    pub rope_embed: CudaFunction,
    pub prefixlm_mask: CudaFunction,
    pub swiglu_ffn: CudaFunction,
    pub gated_attn_output: CudaFunction,
    pub embedding_lookup: CudaFunction,
    pub split_gqkv: CudaFunction,
    pub apply_attn_gate: CudaFunction,
    pub mha: CudaFunction,
    pub mha_decode_fn: CudaFunction,
    pub kv_cache_update: CudaFunction,
    pub broadcast_vec: CudaFunction,
}

impl HrmForwardPass {
    /// Build a new forward-pass engine on the given CUDA device ordinal.
    pub fn new(dev_ordinal: usize) -> CudaResult<Self> {
        let dev = CudaDevice::new(dev_ordinal)?;
        let out_dir = env!("OUT_DIR");

        fn load_kernel(
            dev: &Arc<CudaDevice>,
            out_dir: &str,
            ptx_name: &'static str,
            func_name: &'static str,
        ) -> CudaResult<CudaFunction> {
            let ptx_path = format!("{}/{}.ptx", out_dir, ptx_name);
            let ptx = std::fs::read_to_string(&ptx_path)
                .unwrap_or_else(|_| panic!("PTX not found: {}", ptx_path));
            dev.load_ptx(ptx.into(), ptx_name, &[func_name])?;
            dev.get_func(ptx_name, func_name)
                .ok_or_else(|| HrmError::KernelNotFound(func_name.to_string()))
        }

        let blas = cudarc::cublas::CudaBlas::new(dev.clone())?;

        Ok(Self {
            additive_inject: load_kernel(
                &dev,
                out_dir,
                "additive_inject",
                "additive_inject_kernel_vec8",
            )?,
            rms_norm: load_kernel(&dev, out_dir, "rms_norm", "rms_norm_kernel")?,
            rope_embed: load_kernel(&dev, out_dir, "rope_embed", "rope_embed_kernel")?,
            prefixlm_mask: load_kernel(&dev, out_dir, "prefixlm_mask", "prefixlm_mask_kernel")?,
            swiglu_ffn: load_kernel(&dev, out_dir, "swiglu_ffn", "swiglu_ffn_kernel")?,
            gated_attn_output: load_kernel(
                &dev,
                out_dir,
                "gated_attn_output",
                "gated_attn_output_kernel",
            )?,
            embedding_lookup: load_kernel(
                &dev,
                out_dir,
                "embedding_lookup",
                "embedding_lookup_kernel",
            )?,
            split_gqkv: load_kernel(&dev, out_dir, "split_gqkv", "split_gqkv_kernel")?,
            apply_attn_gate: load_kernel(
                &dev,
                out_dir,
                "apply_attn_gate",
                "apply_attn_gate_kernel",
            )?,
            mha: load_kernel(&dev, out_dir, "mha", "mha_kernel")?,
            mha_decode_fn: load_kernel(&dev, out_dir, "mha_decode", "mha_decode_kernel")?,
            kv_cache_update: load_kernel(
                &dev,
                out_dir,
                "kv_cache_update",
                "kv_cache_update_kernel",
            )?,
            broadcast_vec: load_kernel(&dev, out_dir, "broadcast_vec", "broadcast_vec_kernel")?,
            blas,
            dev,
            config: HrmConfig::default(),
            weights: None,
            scratch: None,
            kv_cache: None,
            z_h_buf: None,
            z_l_buf: None,
            normed_buf: None,
            decode_mask_buf: None,
        })
    }

    /// Allocate a bf16 device buffer of `len` elements, zero-initialized.
    pub fn alloc_zero_bf16(&self, len: usize) -> CudaResult<CudaSlice<bf16>> {
        let mut buf = unsafe { self.dev.alloc::<bf16>(len)? };
        self.dev.memset_zeros(&mut buf)?;
        Ok(buf)
    }

    /// Copy a host bf16 slice to a new device buffer.
    pub fn htod_copy_bf16(&self, host: &[bf16]) -> CudaResult<CudaSlice<bf16>> {
        Ok(self.dev.htod_copy(host.to_vec())?)
    }

    // ------------------------------------------------------------------
    // Ensure scratch buffers are allocated (or reallocated if too small)
    // ------------------------------------------------------------------
    fn ensure_scratch(&mut self, batch_size: usize, seq_len: usize) -> CudaResult<()> {
        let rows = batch_size * seq_len;
        if let Some(ref s) = self.scratch {
            if s.alloc_rows >= rows && s.alloc_seq >= seq_len {
                return Ok(());
            }
        }
        // Drop the old allocation before growing it to avoid a large peak-memory spike.
        self.scratch = None;
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let heads = cfg.num_heads;
        let hdim = cfg.head_dim;

        self.scratch = Some(ScratchBuffers {
            norm_buf: self.alloc_zero_bf16(rows * hidden)?,
            gqkv_buf: self.alloc_zero_bf16(rows * 4 * hidden)?,
            gate_buf: self.alloc_zero_bf16(rows * heads * hdim)?,
            q_buf: self.alloc_zero_bf16(rows * heads * hdim)?,
            k_buf: self.alloc_zero_bf16(rows * heads * hdim)?,
            v_buf: self.alloc_zero_bf16(rows * heads * hdim)?,
            attn_out_buf: self.alloc_zero_bf16(rows * hidden)?,
            o_out_buf: self.alloc_zero_bf16(rows * hidden)?,
            norm2_buf: self.alloc_zero_bf16(rows * hidden)?,
            ffn_gate_buf: self.alloc_zero_bf16(rows * inter)?,
            ffn_up_buf: self.alloc_zero_bf16(rows * inter)?,
            ffn_out_buf: self.alloc_zero_bf16(rows * inter)?,
            down_buf: self.alloc_zero_bf16(rows * hidden)?,
            residual_buf: self.alloc_zero_bf16(rows * hidden)?,
            tmp1: self.alloc_zero_bf16(rows * hidden)?,
            tmp2: self.alloc_zero_bf16(rows * hidden)?,
            mask_buf: self.alloc_zero_bf16(batch_size * seq_len * seq_len)?,
            logits_buf: self.alloc_zero_bf16(cfg.vocab_size)?,
            alloc_rows: rows,
            alloc_seq: seq_len,
        });
        Ok(())
    }

    // ------------------------------------------------------------------
    // Ensure persistent buffers are allocated (or reallocated if too small)
    // ------------------------------------------------------------------
    fn ensure_persistent_bufs(&mut self, batch_size: usize) -> CudaResult<()> {
        let max_seq = self.config.max_seq_len;
        let hidden = self.config.hidden_size;
        let rows = batch_size * max_seq;
        if self.z_h_buf.is_none() {
            self.z_h_buf = Some(self.alloc_zero_bf16(rows * hidden)?);
        }
        if self.z_l_buf.is_none() {
            self.z_l_buf = Some(self.alloc_zero_bf16(rows * hidden)?);
        }
        if self.normed_buf.is_none() {
            self.normed_buf = Some(self.alloc_zero_bf16(rows * hidden)?);
        }
        if self.decode_mask_buf.is_none() {
            self.decode_mask_buf = Some(self.alloc_zero_bf16(batch_size * max_seq)?);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // cuBLAS GEMM: C = A @ B^T
    // A: [m, k] row-major, B: [n, k] row-major, C: [m, n] row-major
    // ------------------------------------------------------------------
    fn matmul<A, B, C>(&self, a: &A, b: &B, c: &mut C, m: i32, n: i32, k: i32) -> CudaResult<()>
    where
        A: DevicePtr<bf16>,
        B: DevicePtr<bf16>,
        C: DevicePtrMut<bf16>,
    {
        use cudarc::cublas::sys as cublas_sys;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cublas_sys::lib()
                .cublasGemmEx(
                    *self.blas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    *b.device_ptr() as *const std::ffi::c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    k,
                    *a.device_ptr() as *const std::ffi::c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    k,
                    &beta as *const f32 as *const std::ffi::c_void,
                    *c.device_ptr_mut() as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    n,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .result()?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Kernel wrappers
    // ------------------------------------------------------------------
    pub fn additive_inject(
        &self,
        z_a: &CudaSlice<bf16>,
        z_b: &CudaSlice<bf16>,
        z_out: &mut CudaSlice<bf16>,
        total_elements: usize,
    ) -> CudaResult<()> {
        let threads = 256usize;
        let vec_elems = (total_elements + 7) / 8;
        let blocks = (vec_elems + threads - 1) / threads;
        unsafe {
            self.additive_inject.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (z_a, z_b, z_out, total_elements as i32),
            )?;
        }
        Ok(())
    }

    pub fn rms_norm(
        &self,
        input: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        rows: usize,
        hidden_size: usize,
    ) -> CudaResult<()> {
        let threads = 256usize;
        let warps = threads / 32;
        let shared = warps * std::mem::size_of::<f32>();
        unsafe {
            self.rms_norm.clone().launch(
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: shared as u32,
                },
                (input, output, hidden_size as i32, self.config.norm_eps),
            )?;
        }
        Ok(())
    }

    pub fn rope_embed(
        &self,
        q: &mut CudaSlice<bf16>,
        k: &mut CudaSlice<bf16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        position_offset: usize,
    ) -> CudaResult<()> {
        let half_dim = head_dim / 2;
        let total = batch_size * num_heads * seq_len * half_dim;
        let threads = 256usize;
        let blocks = (total + threads - 1) / threads;
        unsafe {
            self.rope_embed.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    q,
                    k,
                    batch_size as i32,
                    num_heads as i32,
                    seq_len as i32,
                    head_dim as i32,
                    position_offset as i32,
                    self.config.rope_theta,
                ),
            )?;
        }
        Ok(())
    }

    pub fn prefixlm_mask(
        &self,
        token_type_ids: &CudaSlice<i32>,
        mask: &mut CudaSlice<bf16>,
        batch_size: usize,
        seq_len: usize,
    ) -> CudaResult<()> {
        let total = batch_size * seq_len * seq_len;
        let threads = 256usize;
        let blocks = (total + threads - 1) / threads;
        unsafe {
            self.prefixlm_mask.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (token_type_ids, mask, batch_size as i32, seq_len as i32),
            )?;
        }
        Ok(())
    }

    pub fn swiglu_ffn(
        &self,
        gate: &CudaSlice<bf16>,
        up: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        total_elements: usize,
    ) -> CudaResult<()> {
        let threads = 256usize;
        let vec_elems = (total_elements + 7) / 8;
        let blocks = (vec_elems + threads - 1) / threads;
        unsafe {
            self.swiglu_ffn.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (gate, up, output, total_elements as i32),
            )?;
        }
        Ok(())
    }

    pub fn gated_attn_output(
        &self,
        gate: &CudaSlice<bf16>,
        attn_out: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        total_elements: usize,
    ) -> CudaResult<()> {
        let threads = 256usize;
        let vec_elems = (total_elements + 7) / 8;
        let blocks = (vec_elems + threads - 1) / threads;
        unsafe {
            self.gated_attn_output.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (gate, attn_out, output, total_elements as i32),
            )?;
        }
        Ok(())
    }

    pub fn embedding_lookup(
        &self,
        input_ids: &CudaSlice<u32>,
        embedding_table: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        embedding_scale: f32,
    ) -> CudaResult<()> {
        // Vectorized kernel processes 8 bf16 per thread
        let vec_hidden = hidden_size / 8;
        let total_vec = batch_size * seq_len * vec_hidden;
        let threads = 256usize;
        let blocks = (total_vec + threads - 1) / threads;
        unsafe {
            self.embedding_lookup.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    input_ids,
                    embedding_table,
                    output,
                    batch_size as i32,
                    seq_len as i32,
                    hidden_size as i32,
                    vocab_size as i32,
                    embedding_scale,
                ),
            )?;
        }
        Ok(())
    }

    pub fn split_gqkv(
        &self,
        gqkv: &CudaSlice<bf16>,
        gate: &mut CudaSlice<bf16>,
        q: &mut CudaSlice<bf16>,
        k: &mut CudaSlice<bf16>,
        v: &mut CudaSlice<bf16>,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> CudaResult<()> {
        // Vectorized kernel processes 8 bf16 per thread
        let vec_dim = head_dim / 8;
        let total_vec = batch_size * seq_len * num_heads * vec_dim;
        let threads = 256usize;
        let blocks = (total_vec + threads - 1) / threads;
        unsafe {
            self.split_gqkv.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    gqkv,
                    gate,
                    q,
                    k,
                    v,
                    batch_size as i32,
                    seq_len as i32,
                    num_heads as i32,
                    head_dim as i32,
                ),
            )?;
        }
        Ok(())
    }

    pub fn apply_attn_gate(
        &self,
        gate: &CudaSlice<bf16>,
        attn_out: &mut CudaSlice<bf16>,
        total_elements: usize,
    ) -> CudaResult<()> {
        let threads = 256usize;
        let vec_elems = (total_elements + 7) / 8;
        let blocks = (vec_elems + threads - 1) / threads;
        unsafe {
            self.apply_attn_gate.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (gate, attn_out, total_elements as i32),
            )?;
        }
        Ok(())
    }

    pub fn mha(
        &self,
        q: &CudaSlice<bf16>,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        mask: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> CudaResult<()> {
        let warps = 128usize / 32; // = 4
        let shared = ((head_dim + seq_len + warps) * std::mem::size_of::<f32>()) as u32;
        unsafe {
            self.mha.clone().launch(
                LaunchConfig {
                    grid_dim: ((batch_size * num_heads) as u32, seq_len as u32, 1),
                    block_dim: (128u32, 1, 1),
                    shared_mem_bytes: shared,
                },
                (
                    q,
                    k,
                    v,
                    mask,
                    output,
                    batch_size as i32,
                    num_heads as i32,
                    seq_len as i32,
                    head_dim as i32,
                ),
            )?;
        }
        Ok(())
    }

    pub fn mha_decode(
        &self,
        q: &CudaSlice<bf16>,
        k_cache: &CudaSlice<bf16>,
        v_cache: &CudaSlice<bf16>,
        mask: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        batch_size: usize,
        num_heads: usize,
        kv_len: usize,
        head_dim: usize,
        max_len: usize,
    ) -> CudaResult<()> {
        let threads = 128usize;
        let warps = threads / 32;
        let shared = ((head_dim + kv_len + warps) * std::mem::size_of::<f32>()) as u32;
        unsafe {
            self.mha_decode_fn.clone().launch(
                LaunchConfig {
                    grid_dim: ((batch_size * num_heads) as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: shared,
                },
                (
                    q,
                    k_cache,
                    v_cache,
                    mask,
                    output,
                    batch_size as i32,
                    num_heads as i32,
                    kv_len as i32,
                    head_dim as i32,
                    max_len as i32,
                ),
            )?;
        }
        Ok(())
    }

    pub fn kv_cache_update(
        &self,
        src_k: &CudaSlice<bf16>,
        src_v: &CudaSlice<bf16>,
        dst_k: &mut CudaSlice<bf16>,
        dst_v: &mut CudaSlice<bf16>,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        max_len: usize,
        cached_len: usize,
        is_decode: bool,
    ) -> CudaResult<()> {
        let total_elements = batch_size * num_heads * seq_len * head_dim;
        let threads = 256usize;
        let blocks = ((total_elements + threads - 1) / threads).min(65535).max(1);
        unsafe {
            self.kv_cache_update.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    src_k,
                    src_v,
                    dst_k,
                    dst_v,
                    batch_size as i32,
                    num_heads as i32,
                    seq_len as i32,
                    head_dim as i32,
                    max_len as i32,
                    cached_len as i32,
                    is_decode as i32,
                ),
            )?;
        }
        Ok(())
    }

    pub fn broadcast_vec(
        &self,
        dst: &mut CudaSlice<bf16>,
        src: &CudaSlice<bf16>,
        n_copies: usize,
        vec_len: usize,
    ) -> CudaResult<()> {
        let total = n_copies * vec_len;
        let threads = 256usize;
        let blocks = ((total + threads - 1) / threads).min(65535).max(1);
        unsafe {
            self.broadcast_vec.clone().launch(
                LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (dst, src, n_copies as i32, vec_len as i32),
            )?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Attention (MHA with per-head sigmoid gate, matching Python)
    // Accepts scratch buffers as parameter to avoid borrow conflicts.
    // kv_slot: index into KvCache for this attention call.
    // is_decode: if true, use cached K/V and mha_decode kernel.
    // ------------------------------------------------------------------
    fn attention(
        &self,
        layer: &LayerWeights,
        input: &CudaSlice<bf16>,
        mask: &CudaSlice<bf16>,
        scratch: &mut ScratchBuffers,
        kv_cache: Option<*mut KvCache>,
        kv_slot: usize,
        is_decode: bool,
        batch_size: usize,
        seq_len: usize,
    ) -> CudaResult<()> {
        let cfg = &self.config;
        let heads = cfg.num_heads;
        let hdim = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let rows = batch_size * seq_len;
        let total_hidden = rows * hidden;
        let total_heads = rows * heads * hdim;

        // gqkv_proj: [rows, hidden] @ [4*hidden, hidden]^T → [rows, 4*hidden]
        self.matmul(
            input,
            &layer.gqkv_proj,
            &mut scratch.gqkv_buf,
            rows as i32,
            (4 * hidden) as i32,
            hidden as i32,
        )?;

        // Split into gate, q, k, v each [rows, heads*hdim]
        self.split_gqkv(
            &scratch.gqkv_buf,
            &mut scratch.gate_buf,
            &mut scratch.q_buf,
            &mut scratch.k_buf,
            &mut scratch.v_buf,
            batch_size,
            seq_len,
            heads,
            hdim,
        )?;

        // Apply RoPE in-place to Q and K
        let position_offset = if is_decode {
            kv_cache
                .map(|cache_ptr| unsafe { (&*cache_ptr).cached_len })
                .unwrap_or(0)
        } else {
            0
        };
        self.rope_embed(
            &mut scratch.q_buf,
            &mut scratch.k_buf,
            batch_size,
            heads,
            seq_len,
            hdim,
            position_offset,
        )?;

        if is_decode {
            // Decode path: append K/V to cache, attend against full cache
            if let Some(cache_ptr) = kv_cache {
                let cache = unsafe { &mut *cache_ptr };
                let kv_len = cache.cached_len + 1;

                // Batched copy of new K/V into cache at position cached_len
                self.kv_cache_update(
                    &scratch.k_buf,
                    &scratch.v_buf,
                    &mut cache.k_cache[kv_slot],
                    &mut cache.v_cache[kv_slot],
                    batch_size,
                    heads,
                    seq_len,
                    hdim,
                    cache.max_len,
                    cache.cached_len,
                    true,
                )?;

                // Run MHA decode against full cache
                self.mha_decode(
                    &scratch.q_buf,
                    &cache.k_cache[kv_slot],
                    &cache.v_cache[kv_slot],
                    mask,
                    &mut scratch.attn_out_buf,
                    batch_size,
                    heads,
                    kv_len,
                    hdim,
                    cache.max_len,
                )?;
            } else {
                // Decode without KV cache: run single-token MHA against current K/V
                self.mha(
                    &scratch.q_buf,
                    &scratch.k_buf,
                    &scratch.v_buf,
                    mask,
                    &mut scratch.attn_out_buf,
                    batch_size,
                    heads,
                    seq_len,
                    hdim,
                )?;
            }
        } else {
            // Prefill path: run normal MHA
            self.mha(
                &scratch.q_buf,
                &scratch.k_buf,
                &scratch.v_buf,
                mask,
                &mut scratch.attn_out_buf,
                batch_size,
                heads,
                seq_len,
                hdim,
            )?;

            // Batched copy K/V into cache for future decode
            if let Some(cache_ptr) = kv_cache {
                let cache = unsafe { &mut *cache_ptr };
                self.kv_cache_update(
                    &scratch.k_buf,
                    &scratch.v_buf,
                    &mut cache.k_cache[kv_slot],
                    &mut cache.v_cache[kv_slot],
                    batch_size,
                    heads,
                    seq_len,
                    hdim,
                    cache.max_len,
                    0,
                    false,
                )?;
            }
        }

        // Apply sigmoid(gate) * attn_output elementwise
        self.apply_attn_gate(&scratch.gate_buf, &mut scratch.attn_out_buf, total_heads)?;

        // o_proj: [rows, heads*hdim] @ [hidden, hidden]^T → [rows, hidden]
        self.matmul(
            &scratch.attn_out_buf,
            &layer.o_proj,
            &mut scratch.o_out_buf,
            rows as i32,
            hidden as i32,
            hidden as i32,
        )?;

        // Copy o_out_buf back to attn_out_buf (which serves as the attention output)
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *scratch.attn_out_buf.device_ptr_mut(),
                *scratch.o_out_buf.device_ptr(),
                std::mem::size_of::<bf16>() * total_hidden,
                *self.dev.cu_stream(),
            )?;
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Single Transformer layer (Pre-Norm)
    // ------------------------------------------------------------------
    fn transformer_layer(
        &self,
        layer: &LayerWeights,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        mask: &CudaSlice<bf16>,
        scratch: &mut ScratchBuffers,
        kv_cache: Option<*mut KvCache>,
        kv_slot: usize,
        is_decode: bool,
        batch_size: usize,
        seq_len: usize,
    ) -> CudaResult<()> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let rows = batch_size * seq_len;
        let total_hidden = rows * hidden;
        let total_inter = rows * intermediate;

        // --- Attention branch ---
        self.rms_norm(x, &mut scratch.norm_buf, rows, hidden)?;
        let norm_buf = scratch.norm_buf.clone();
        self.attention(
            layer, &norm_buf, mask, scratch, kv_cache, kv_slot, is_decode, batch_size, seq_len,
        )?;

        // Residual: out = x + attn_out_buf
        self.additive_inject(x, &scratch.attn_out_buf, out, total_hidden)?;

        // --- FFN branch ---
        self.rms_norm(out, &mut scratch.norm2_buf, rows, hidden)?;

        self.matmul(
            &scratch.norm2_buf,
            &layer.gate_proj,
            &mut scratch.ffn_gate_buf,
            rows as i32,
            intermediate as i32,
            hidden as i32,
        )?;
        self.matmul(
            &scratch.norm2_buf,
            &layer.up_proj,
            &mut scratch.ffn_up_buf,
            rows as i32,
            intermediate as i32,
            hidden as i32,
        )?;

        self.swiglu_ffn(
            &scratch.ffn_gate_buf,
            &scratch.ffn_up_buf,
            &mut scratch.ffn_out_buf,
            total_inter,
        )?;

        self.matmul(
            &scratch.ffn_out_buf,
            &layer.down_proj,
            &mut scratch.down_buf,
            rows as i32,
            hidden as i32,
            intermediate as i32,
        )?;

        // Residual: out = out + down
        self.additive_inject(
            out,
            &scratch.down_buf,
            &mut scratch.residual_buf,
            total_hidden,
        )?;
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *out.device_ptr_mut(),
                *scratch.residual_buf.device_ptr(),
                std::mem::size_of::<bf16>() * total_hidden,
                *self.dev.cu_stream(),
            )?;
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Full Transformer module (H or L), running all layers
    // ------------------------------------------------------------------
    fn transformer_module(
        &self,
        module: &ModuleWeights,
        input: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        mask: &CudaSlice<bf16>,
        scratch: &mut ScratchBuffers,
        kv_cache: Option<*mut KvCache>,
        kv_slot_base: usize,
        is_decode: bool,
        batch_size: usize,
        seq_len: usize,
    ) -> CudaResult<()> {
        let total_hidden = batch_size * seq_len * self.config.hidden_size;

        // First layer reads from input
        self.transformer_layer(
            &module.layers[0],
            input,
            output,
            mask,
            scratch,
            kv_cache,
            kv_slot_base,
            is_decode,
            batch_size,
            seq_len,
        )?;

        // Remaining layers read from output
        for (i, layer) in module.layers[1..].iter().enumerate() {
            // Copy output -> tmp1 to use as input
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    *scratch.tmp1.device_ptr_mut(),
                    *output.device_ptr(),
                    std::mem::size_of::<bf16>() * total_hidden,
                    *self.dev.cu_stream(),
                )?;
            }
            let tmp1 = scratch.tmp1.clone();
            self.transformer_layer(
                layer,
                &tmp1,
                output,
                mask,
                scratch,
                kv_cache,
                kv_slot_base + i + 1,
                is_decode,
                batch_size,
                seq_len,
            )?;
        }

        self.rms_norm(
            output,
            &mut scratch.tmp1,
            batch_size * seq_len,
            self.config.hidden_size,
        )?;
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *output.device_ptr_mut(),
                *scratch.tmp1.device_ptr(),
                std::mem::size_of::<bf16>() * total_hidden,
                *self.dev.cu_stream(),
            )?;
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Full recurrent forward pass (original, kept for compatibility)
    // ------------------------------------------------------------------
    pub fn forward(
        &mut self,
        input_ids: &[u32],
        token_type_ids: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) -> CudaResult<CudaSlice<bf16>> {
        self.forward_prefill(input_ids, token_type_ids, batch_size, seq_len)
    }

    // ------------------------------------------------------------------
    // Prefill: full forward pass, populates KV cache, returns last-token logits
    // ------------------------------------------------------------------
    pub fn forward_prefill(
        &mut self,
        input_ids: &[u32],
        token_type_ids: &[i32],
        batch_size: usize,
        seq_len: usize,
    ) -> CudaResult<CudaSlice<bf16>> {
        let max_len = self.config.max_seq_len;
        self.forward_prefill_with_capacity(input_ids, token_type_ids, batch_size, seq_len, max_len)
    }

    fn forward_prefill_with_capacity(
        &mut self,
        input_ids: &[u32],
        token_type_ids: &[i32],
        batch_size: usize,
        seq_len: usize,
        cache_capacity: usize,
    ) -> CudaResult<CudaSlice<bf16>> {
        if batch_size != 1 {
            return Err(HrmError::InvalidInput(
                "only batch_size=1 is currently supported".to_string(),
            ));
        }
        if seq_len == 0 {
            return Err(HrmError::InvalidInput(
                "prompt contains no tokens".to_string(),
            ));
        }
        if input_ids.len() != batch_size * seq_len || token_type_ids.len() != batch_size * seq_len {
            return Err(HrmError::InvalidInput(
                "input and token type lengths must match batch_size * seq_len".to_string(),
            ));
        }
        if cache_capacity < seq_len || cache_capacity > self.config.max_seq_len {
            return Err(HrmError::InvalidInput(format!(
                "cache capacity must be between prompt length {} and model limit {}",
                seq_len, self.config.max_seq_len
            )));
        }

        // Ensure scratch and persistent buffers
        self.ensure_scratch(batch_size, seq_len)?;
        self.ensure_persistent_bufs(batch_size)?;

        // Take buffers out to avoid borrow conflicts
        let mut scratch = self.scratch.take().unwrap();
        let mut z_h = self.z_h_buf.take().unwrap();
        let mut z_l = self.z_l_buf.take().unwrap();
        let normed = self.normed_buf.take().unwrap();

        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| HrmError::KernelNotFound("weights not loaded".to_string()))?;
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let total_hidden = batch_size * seq_len * hidden;
        let rows = batch_size * seq_len;

        // Drop old KV cache first to avoid peak memory doubling, then allocate new
        self.kv_cache = None;
        self.kv_cache = Some(KvCache::new(&self.dev, cfg, batch_size, cache_capacity)?);

        // Embedding lookup (with scaling from ScaledEmbeddingInit)
        let input_ids_dev = self.dev.htod_copy(input_ids.to_vec())?;
        self.embedding_lookup(
            &input_ids_dev,
            &weights.embed_tokens,
            &mut z_h,
            batch_size,
            seq_len,
            hidden,
            cfg.vocab_size,
            cfg.embedding_scale,
        )?;

        // Initialize z_L from z_l_init parameter by broadcasting to every position
        self.broadcast_vec(&mut z_l, &weights.z_l_init, batch_size * seq_len, hidden)?;

        // PrefixLM mask
        let token_type_ids_dev = self.dev.htod_copy(token_type_ids.to_vec())?;
        self.prefixlm_mask(
            &token_type_ids_dev,
            &mut scratch.mask_buf,
            batch_size,
            seq_len,
        )?;

        // Recurrent loop (HRM)
        let n_layers = cfg.n_layers;
        let mut kv_slot_counter = 0usize;
        for _h in 0..cfg.h_cycles {
            for _l in 0..cfg.l_cycles {
                // z_L = L_module(z_L + z_H)
                self.additive_inject(&z_l, &z_h, &mut scratch.tmp1, total_hidden)?;
                // Copy tmp1 to tmp2 as module input (tmp1 may be reused by module internals)
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        *scratch.tmp2.device_ptr_mut(),
                        *scratch.tmp1.device_ptr(),
                        std::mem::size_of::<bf16>() * total_hidden,
                        *self.dev.cu_stream(),
                    )?;
                }
                let kv_base = kv_slot_counter;
                kv_slot_counter += n_layers;
                let kv_ptr = self.kv_cache.as_mut().map(|c| c as *mut KvCache);
                let tmp2 = scratch.tmp2.clone();
                let mask_buf = scratch.mask_buf.clone();
                self.transformer_module(
                    &weights.l_module,
                    &tmp2,
                    &mut z_l,
                    &mask_buf,
                    &mut scratch,
                    kv_ptr,
                    kv_base,
                    false,
                    batch_size,
                    seq_len,
                )?;
            }
            // z_H = H_module(z_H + z_L)
            self.additive_inject(&z_h, &z_l, &mut scratch.tmp1, total_hidden)?;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    *scratch.tmp2.device_ptr_mut(),
                    *scratch.tmp1.device_ptr(),
                    std::mem::size_of::<bf16>() * total_hidden,
                    *self.dev.cu_stream(),
                )?;
            }
            let kv_base = kv_slot_counter;
            kv_slot_counter += n_layers;
            let kv_ptr = self.kv_cache.as_mut().map(|c| c as *mut KvCache);
            let tmp2 = scratch.tmp2.clone();
            let mask_buf = scratch.mask_buf.clone();
            self.transformer_module(
                &weights.h_module,
                &tmp2,
                &mut z_h,
                &mask_buf,
                &mut scratch,
                kv_ptr,
                kv_base,
                false,
                batch_size,
                seq_len,
            )?;
        }

        // LM head → logits [batch, seq, vocab_size]
        // Generation only needs the final prompt position. Avoid materializing
        // [seq_len, vocab_size] logits, which can consume hundreds of MiB.
        let last_hidden_offset = (rows - 1) * hidden;
        let last_hidden = z_h.slice(last_hidden_offset..last_hidden_offset + hidden);
        self.matmul(
            &last_hidden,
            &weights.lm_head,
            &mut scratch.logits_buf,
            1,
            cfg.vocab_size as i32,
            hidden as i32,
        )?;

        // Copy logits to a new buffer to return (buffers will be put back)
        let mut logits = self.alloc_zero_bf16(cfg.vocab_size)?;
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *logits.device_ptr_mut(),
                *scratch.logits_buf.device_ptr(),
                std::mem::size_of::<bf16>() * cfg.vocab_size,
                *self.dev.cu_stream(),
            )?;
        }

        // Update KV cache length
        if let Some(ref mut cache) = self.kv_cache {
            cache.cached_len = seq_len;
        }

        // Put buffers back
        self.scratch = Some(scratch);
        self.z_h_buf = Some(z_h);
        self.z_l_buf = Some(z_l);
        self.normed_buf = Some(normed);

        Ok(logits)
    }

    // ------------------------------------------------------------------
    // Decode: process single new token using KV cache
    // ------------------------------------------------------------------
    pub fn forward_decode(
        &mut self,
        new_token: &[u32],
        _new_type: &[i32],
        batch_size: usize,
    ) -> CudaResult<CudaSlice<bf16>> {
        if batch_size != 1 || new_token.len() != 1 {
            return Err(HrmError::InvalidInput(
                "decode currently requires exactly one token with batch_size=1".to_string(),
            ));
        }
        let seq_len = 1usize; // single new token

        // Ensure scratch and persistent buffers (decode uses seq_len=1)
        self.ensure_scratch(batch_size, seq_len)?;
        self.ensure_persistent_bufs(batch_size)?;

        let mut scratch = self.scratch.take().unwrap();
        let mut z_h = self.z_h_buf.take().unwrap();
        let mut z_l = self.z_l_buf.take().unwrap();
        let normed = self.normed_buf.take().unwrap();

        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| HrmError::KernelNotFound("weights not loaded".to_string()))?;
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let total_hidden = batch_size * seq_len * hidden;
        let rows = batch_size * seq_len;

        let kv_len = self.kv_cache.as_ref().map_or(0, |c| c.cached_len) + 1;
        if let Some(cache) = self.kv_cache.as_ref() {
            if kv_len > cache.max_len {
                return Err(HrmError::InvalidInput(format!(
                    "generation would exceed the allocated context length of {} tokens",
                    cache.max_len
                )));
            }
        }

        // Embedding lookup for new token
        let input_ids_dev = self.dev.htod_copy(new_token.to_vec())?;
        self.embedding_lookup(
            &input_ids_dev,
            &weights.embed_tokens,
            &mut z_h,
            batch_size,
            seq_len,
            hidden,
            cfg.vocab_size,
            cfg.embedding_scale,
        )?;

        // Initialize z_L from z_l_init
        self.broadcast_vec(&mut z_l, &weights.z_l_init, batch_size, hidden)?;

        // Decode mask: take persistent buffer out to avoid borrow conflicts
        let mut decode_mask = self.decode_mask_buf.take().unwrap();
        // Zero the full buffer (sized for max_seq, already zeroed at allocation, but re-zero to be safe)
        self.dev.memset_zeros(&mut decode_mask)?;

        // Recurrent loop (HRM) with decode
        let n_layers = cfg.n_layers;
        let mut kv_slot_counter = 0usize;
        for _h in 0..cfg.h_cycles {
            for _l in 0..cfg.l_cycles {
                self.additive_inject(&z_l, &z_h, &mut scratch.tmp1, total_hidden)?;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        *scratch.tmp2.device_ptr_mut(),
                        *scratch.tmp1.device_ptr(),
                        std::mem::size_of::<bf16>() * total_hidden,
                        *self.dev.cu_stream(),
                    )?;
                }
                let kv_base = kv_slot_counter;
                kv_slot_counter += n_layers;
                let kv_ptr = self.kv_cache.as_mut().map(|c| c as *mut KvCache);
                let tmp2 = scratch.tmp2.clone();
                self.transformer_module(
                    &weights.l_module,
                    &tmp2,
                    &mut z_l,
                    &decode_mask,
                    &mut scratch,
                    kv_ptr,
                    kv_base,
                    true,
                    batch_size,
                    seq_len,
                )?;
            }
            self.additive_inject(&z_h, &z_l, &mut scratch.tmp1, total_hidden)?;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    *scratch.tmp2.device_ptr_mut(),
                    *scratch.tmp1.device_ptr(),
                    std::mem::size_of::<bf16>() * total_hidden,
                    *self.dev.cu_stream(),
                )?;
            }
            let kv_base = kv_slot_counter;
            kv_slot_counter += n_layers;
            let kv_ptr = self.kv_cache.as_mut().map(|c| c as *mut KvCache);
            let tmp2 = scratch.tmp2.clone();
            self.transformer_module(
                &weights.h_module,
                &tmp2,
                &mut z_h,
                &decode_mask,
                &mut scratch,
                kv_ptr,
                kv_base,
                true,
                batch_size,
                seq_len,
            )?;
        }

        // LM head
        self.matmul(
            &z_h,
            &weights.lm_head,
            &mut scratch.logits_buf,
            rows as i32,
            cfg.vocab_size as i32,
            hidden as i32,
        )?;

        let mut logits = self.alloc_zero_bf16(rows * cfg.vocab_size)?;
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *logits.device_ptr_mut(),
                *scratch.logits_buf.device_ptr(),
                std::mem::size_of::<bf16>() * rows * cfg.vocab_size,
                *self.dev.cu_stream(),
            )?;
        }

        // Update KV cache length
        if let Some(ref mut cache) = self.kv_cache {
            cache.cached_len = kv_len;
        }

        // Put buffers back
        self.scratch = Some(scratch);
        self.z_h_buf = Some(z_h);
        self.z_l_buf = Some(z_l);
        self.normed_buf = Some(normed);
        self.decode_mask_buf = Some(decode_mask);

        Ok(logits)
    }
}

// ------------------------------------------------------------------
// High-level model: tokenizer + weights + generation
// ------------------------------------------------------------------
pub struct HrmTextModel {
    pub forward_pass: HrmForwardPass,
    pub tokenizer: tokenizers::Tokenizer,
    pub eos_token_id: u32,
}

#[derive(Clone, Debug)]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl Sampler {
    pub fn sample(&self, logits: &[bf16], past_tokens: &[u32], vocab_size: usize) -> u32 {
        let vocab_size = vocab_size.min(logits.len());
        if vocab_size == 0 {
            return 0;
        }

        let greedy = || {
            let mut best_id = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for i in 0..vocab_size {
                let v = logits[i].to_f32();
                if v.is_finite() && v > best_val {
                    best_val = v;
                    best_id = i;
                }
            }
            best_id as u32
        };

        if !self.temperature.is_finite() || self.temperature <= 1e-5 {
            return greedy();
        }

        let mut probs: Vec<(usize, f32)> = (0..vocab_size)
            .map(|i| {
                let value = logits[i].to_f32();
                (
                    i,
                    if value.is_finite() {
                        value
                    } else {
                        f32::NEG_INFINITY
                    },
                )
            })
            .collect();

        if self.temperature != 1.0 {
            for (_, p) in &mut probs {
                *p /= self.temperature;
            }
        }

        if self.repetition_penalty.is_finite()
            && self.repetition_penalty > 0.0
            && self.repetition_penalty != 1.0
        {
            let unique_tokens: std::collections::HashSet<u32> =
                past_tokens.iter().copied().collect();
            for token in unique_tokens {
                let t = token as usize;
                if t < vocab_size {
                    let (_, p) = &mut probs[t];
                    if *p > 0.0 {
                        *p /= self.repetition_penalty;
                    } else {
                        *p *= self.repetition_penalty;
                    }
                }
            }
        }

        let max_logit = probs
            .iter()
            .map(|(_, p)| *p)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (_, p) in &mut probs {
            *p = (*p - max_logit).exp();
            sum += *p;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return greedy();
        }
        for (_, p) in &mut probs {
            *p /= sum;
        }

        let mut filtered: Vec<(usize, f32)>;
        if self.top_k > 0 && self.top_k < vocab_size {
            let mut sorted = probs;
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let k = self.top_k.min(vocab_size);
            filtered = sorted.into_iter().take(k).collect();
        } else {
            filtered = probs;
        }

        let top_p = if self.top_p.is_finite() {
            self.top_p.clamp(0.0, 1.0)
        } else {
            1.0
        };
        if top_p < 1.0 && !filtered.is_empty() {
            filtered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cumsum = 0.0f32;
            let mut cutoff = filtered.len();
            for (i, (_, p)) in filtered.iter().enumerate() {
                cumsum += p;
                if cumsum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            filtered.truncate(cutoff);
        }

        let sum: f32 = filtered.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in &mut filtered {
                *p /= sum;
            }
        }

        let r = fastrand::f32();
        let mut cumsum = 0.0f32;
        for &(id, p) in &filtered {
            cumsum += p;
            if r < cumsum {
                return id as u32;
            }
        }
        filtered.last().map(|(id, _)| *id as u32).unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PromptCondition {
    Direct,
    #[default]
    Reasoning,
}

impl PromptCondition {
    pub fn tokens(&self) -> &'static str {
        match self {
            Self::Direct => "<|object_ref_start|>",
            Self::Reasoning => "<|quad_end|><|object_ref_end|>",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChatSession {
    pub messages: Vec<ChatMessage>,
    pub system_prompt: Option<String>,
    pub sampler: Sampler,
    pub max_tokens: usize,
    pub use_history: bool,
    pub condition: PromptCondition,
}

impl ChatSession {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: None,
            sampler: Sampler::default(),
            max_tokens: 256,
            use_history: false,
            condition: PromptCondition::default(),
        }
    }

    pub fn with_sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = Some(prompt.into());
    }

    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: content.into(),
        });
    }

    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: content.into(),
        });
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn build_prompt(&self) -> String {
        let latest_user = self
            .messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .map(|message| message.content.as_str())
            .unwrap_or("");

        let body = if self.use_history && self.messages.len() > 1 {
            let mut parts = Vec::new();
            if let Some(ref system_prompt) = self.system_prompt {
                if !system_prompt.trim().is_empty() {
                    parts.push(format!("System: {}", system_prompt.trim()));
                }
            }
            parts.push(
                "Use the conversation transcript to answer the final user message. \
Reply with only the answer."
                    .to_string(),
            );
            for message in self
                .messages
                .iter()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                let role = if message.role == "assistant" {
                    "Assistant"
                } else {
                    "User"
                };
                parts.push(format!("{}: {}", role, message.content.trim()));
            }
            parts.push("Assistant:".to_string());
            parts.join("\n\n")
        } else {
            let mut parts = Vec::new();
            if let Some(ref system_prompt) = self.system_prompt {
                if !system_prompt.trim().is_empty() {
                    parts.push(system_prompt.trim().to_string());
                }
            }
            parts.push(latest_user.trim().to_string());
            parts.join("\n\n")
        };

        format!("<|im_start|>{}{}<|im_end|>", self.condition.tokens(), body)
    }
}

impl HrmTextModel {
    /// Download from HuggingFace Hub (if not cached) and load.
    pub fn from_hf(repo_id: &str, cache_dir: &str, dev_ordinal: usize) -> CudaResult<Self> {
        HrmForwardPass::download_from_hf(repo_id, cache_dir)?;
        Self::from_dir(cache_dir, dev_ordinal)
    }

    /// Load from a local directory containing `model.safetensors`, `tokenizer.json`, and `config.json`.
    pub fn from_dir(dir: &str, dev_ordinal: usize) -> CudaResult<Self> {
        let mut fwd = HrmForwardPass::new(dev_ordinal)?;

        // Load tokenizer
        let tok_path = format!("{}/tokenizer.json", dir);
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| HrmError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        // Load config
        let config_path = format!("{}/config.json", dir);
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

        let eos_token_id = config_json
            .get("eos_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as u32;

        // Override config from HF config
        let hidden = config_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as usize;
        let init_std = 1.0 / (hidden as f32).sqrt();
        fwd.config = HrmConfig {
            hidden_size: hidden,
            num_heads: config_json
                .get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            head_dim: config_json
                .get("head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(128) as usize,
            intermediate_size: config_json
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(2816) as usize,
            h_cycles: config_json
                .get("H_cycles")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize,
            l_cycles: config_json
                .get("L_cycles")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize,
            n_layers: config_json
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            max_seq_len: config_json
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize,
            vocab_size: config_json
                .get("vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(32000) as usize,
            rope_theta: config_json
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32,
            norm_eps: config_json
                .get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            init_std,
            embedding_scale: config_json
                .get("embedding_scale")
                .and_then(|v| v.as_f64())
                .unwrap_or((1.0 / init_std) as f64) as f32,
        };
        if fwd.config.num_heads * fwd.config.head_dim != fwd.config.hidden_size {
            return Err(HrmError::InvalidInput(format!(
                "unsupported model config: num_attention_heads ({}) * head_dim ({}) must equal hidden_size ({})",
                fwd.config.num_heads, fwd.config.head_dim, fwd.config.hidden_size
            )));
        }
        if fwd.config.hidden_size % 8 != 0 || fwd.config.head_dim % 8 != 0 {
            return Err(HrmError::InvalidInput(
                "unsupported model config: hidden_size and head_dim must be divisible by 8"
                    .to_string(),
            ));
        }

        // Load weights from safetensors
        let st_path = format!("{}/model.safetensors", dir);
        if !std::path::Path::new(&st_path).exists() {
            return Err(HrmError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing model weights: {}", st_path),
            )));
        }
        fwd.load_safetensors(&st_path)?;

        Ok(Self {
            forward_pass: fwd,
            tokenizer,
            eos_token_id,
        })
    }

    /// Tokenize a prompt and generate up to `max_new_tokens` tokens.
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> CudaResult<String> {
        self.generate_with_sampler(prompt, max_new_tokens, &Sampler::default())
    }

    pub fn generate_with_sampler(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sampler: &Sampler,
    ) -> CudaResult<String> {
        let mut reply = String::new();
        self.generate_streaming(prompt, max_new_tokens, sampler, |token| {
            reply.push_str(token);
        })?;
        Ok(reply)
    }

    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        sampler: &Sampler,
        mut on_token: impl FnMut(&str),
    ) -> CudaResult<String> {
        if max_new_tokens == 0 {
            return Ok(String::new());
        }

        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| HrmError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = input_ids.len();
        let max_seq_len = self.forward_pass.config.max_seq_len;
        if seq_len == 0 {
            return Err(HrmError::InvalidInput(
                "prompt contains no tokens".to_string(),
            ));
        }
        if seq_len >= max_seq_len {
            return Err(HrmError::InvalidInput(format!(
                "prompt is {} tokens, but the model limit is {}",
                seq_len, max_seq_len
            )));
        }

        let generation_budget = max_new_tokens.min(max_seq_len - seq_len);
        let token_type_ids: Vec<i32> = vec![1i32; seq_len];
        let logits_dev = self.forward_pass.forward_prefill_with_capacity(
            &input_ids,
            &token_type_ids,
            1,
            seq_len,
            seq_len + generation_budget,
        )?;

        let logits_host: Vec<bf16> = self.forward_pass.dev.sync_reclaim(logits_dev)?;
        let vocab_size = self.forward_pass.config.vocab_size;
        let mut best_id = sampler.sample(&logits_host[..vocab_size], &input_ids, vocab_size);

        let mut generated_ids = Vec::with_capacity(generation_budget);
        let mut reply = String::new();
        for generated_index in 0..generation_budget {
            if best_id == self.eos_token_id {
                break;
            }

            input_ids.push(best_id);
            generated_ids.push(best_id);
            let decoded = self.tokenizer.decode(&generated_ids, true)
                .map_err(|e| HrmError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
            emit_stable_text(&decoded, &mut reply, &mut on_token);

            if generated_index + 1 == generation_budget {
                break;
            }

            let new_token = [best_id];
            let new_type = [0i32];
            let logits_dev = self.forward_pass.forward_decode(&new_token, &new_type, 1)?;

            let logits_host: Vec<bf16> = self.forward_pass.dev.sync_reclaim(logits_dev)?;
            best_id = sampler.sample(&logits_host[..vocab_size], &input_ids, vocab_size);
        }

        Ok(reply)
    }
}

fn emit_stable_text(decoded: &str, emitted: &mut String, on_text: &mut impl FnMut(&str)) {
    if !decoded.starts_with(emitted.as_str()) {
        return;
    }

    // Byte-level tokenizers may temporarily decode an incomplete UTF-8 sequence
    // as U+FFFD. Hold that suffix until later tokens make it valid.
    let stable_end = decoded.find('\u{fffd}').unwrap_or(decoded.len());
    if stable_end > emitted.len() {
        let text = &decoded[emitted.len()..stable_end];
        on_text(text);
        emitted.push_str(text);
    }
}

// ------------------------------------------------------------------
// Weight loading helpers
// ------------------------------------------------------------------
impl HrmForwardPass {
    /// Convert raw safetensors bytes to a bf16 host Vec.
    fn raw_to_bf16(&self, dtype: safetensors::Dtype, raw: &[u8]) -> Vec<bf16> {
        match dtype {
            safetensors::Dtype::BF16 => {
                assert_eq!(raw.len() % 2, 0);
                raw.chunks_exact(2)
                    .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
                    .collect()
            }
            safetensors::Dtype::F32 => {
                assert_eq!(raw.len() % 4, 0);
                raw.chunks_exact(4)
                    .map(|bytes| {
                        bf16::from_f32(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    })
                    .collect()
            }
            safetensors::Dtype::F16 => {
                assert_eq!(raw.len() % 2, 0);
                raw.chunks_exact(2)
                    .map(|bytes| {
                        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                        bf16::from_f32(half::f16::from_bits(bits).to_f32())
                    })
                    .collect()
            }
            _ => panic!("unsupported dtype"),
        }
    }

    /// Try to get a tensor by exact name, or return a descriptive error.
    fn get_tensor(map: &mut std::collections::HashMap<String, Vec<bf16>>, name: &str) -> Vec<bf16> {
        map.remove(name).unwrap_or_else(|| {
            let available: Vec<_> = map
                .keys()
                .filter(|k| k.contains("layer") || k.contains("embed") || k.contains("lm_head"))
                .cloned()
                .collect();
            panic!("missing tensor: {}. Available keys: {:?}", name, available)
        })
    }

    /// Load bf16/f32 tensors from a single safetensors file into `self.weights`.
    pub fn load_safetensors(&mut self, path: &str) -> CudaResult<()> {
        use std::collections::HashMap;

        let data = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| HrmError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        // Collect all converted tensors into a map
        let mut tensor_map: HashMap<String, Vec<bf16>> = HashMap::new();
        for (name, view) in tensors.tensors() {
            let host_bf16 = self.raw_to_bf16(view.dtype(), view.data());
            tensor_map.insert(name.to_string(), host_bf16);
        }

        let n_layers = self.config.n_layers;
        let intermediate = self.config.intermediate_size;
        let hidden = self.config.hidden_size;

        // Detect naming convention: some HF checkpoints use .attn./.mlp. prefixes, others flatten
        let has_attn_prefix =
            tensor_map.contains_key(&format!("model.H_module.layers.0.attn.gqkv_proj.weight"));

        // Helper to build a ModuleWeights from module prefix
        let mut build_module = |prefix: &str| -> CudaResult<ModuleWeights> {
            let mut layers = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                let (gqkv_name, o_name, gate_up_name, down_name) = if has_attn_prefix {
                    (
                        format!("{}.{}.attn.gqkv_proj.weight", prefix, i),
                        format!("{}.{}.attn.o_proj.weight", prefix, i),
                        format!("{}.{}.mlp.gate_up_proj.weight", prefix, i),
                        format!("{}.{}.mlp.down_proj.weight", prefix, i),
                    )
                } else {
                    (
                        format!("{}.{}.gqkv_proj.weight", prefix, i),
                        format!("{}.{}.o_proj.weight", prefix, i),
                        format!("{}.{}.gate_up_proj.weight", prefix, i),
                        format!("{}.{}.down_proj.weight", prefix, i),
                    )
                };

                let gqkv = self
                    .dev
                    .htod_copy(Self::get_tensor(&mut tensor_map, &gqkv_name))?;
                let o_proj = self
                    .dev
                    .htod_copy(Self::get_tensor(&mut tensor_map, &o_name))?;

                let gate_up = Self::get_tensor(&mut tensor_map, &gate_up_name);
                // gate_up shape is [2*intermediate, hidden]; split into two halves
                let split_point = intermediate * hidden;
                let gate_host = gate_up[..split_point].to_vec();
                let up_host = gate_up[split_point..].to_vec();
                let gate_proj = self.dev.htod_copy(gate_host)?;
                let up_proj = self.dev.htod_copy(up_host)?;

                let down_proj = self
                    .dev
                    .htod_copy(Self::get_tensor(&mut tensor_map, &down_name))?;

                layers.push(LayerWeights {
                    gqkv_proj: gqkv,
                    o_proj,
                    gate_proj,
                    up_proj,
                    down_proj,
                });
            }
            Ok(ModuleWeights { layers })
        };

        let h_module = build_module("model.H_module.layers")?;
        let l_module = build_module("model.L_module.layers")?;

        let embed_tokens = self.dev.htod_copy(Self::get_tensor(
            &mut tensor_map,
            "model.embed_tokens.weight",
        ))?;

        let lm_head_key = if tensor_map.contains_key("lm_head.weight") {
            "lm_head.weight"
        } else {
            "model.lm_head.weight"
        };
        let lm_head = self
            .dev
            .htod_copy(Self::get_tensor(&mut tensor_map, lm_head_key))?;

        let z_l_init = self
            .dev
            .htod_copy(Self::get_tensor(&mut tensor_map, "model.z_L_init"))?;

        self.weights = Some(ModelWeights {
            h_module,
            l_module,
            embed_tokens,
            lm_head,
            z_l_init,
        });

        Ok(())
    }

    /// Download model.safetensors + config.json + tokenizer.json from HuggingFace Hub.
    pub fn download_from_hf(repo_id: &str, local_dir: &str) -> CudaResult<()> {
        use std::io::Write;
        let base = format!("https://huggingface.co/{}/resolve/main", repo_id);

        for file in &["model.safetensors", "config.json", "tokenizer.json"] {
            let url = format!("{}/{}", base, file);
            let out_path = format!("{}/{}", local_dir, file);
            if std::path::Path::new(&out_path).exists() {
                continue;
            }
            std::fs::create_dir_all(local_dir)?;
            let client = reqwest::blocking::Client::builder()
                .timeout(None)
                .build()
                .map_err(|e| HrmError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            let resp = client
                .get(&url)
                .send()
                .map_err(|e| HrmError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            if !resp.status().is_success() {
                return Err(HrmError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to download {}: HTTP {}", file, resp.status()),
                )));
            }
            let bytes = resp
                .bytes()
                .map_err(|e| HrmError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            let mut f = std::fs::File::create(&out_path)?;
            f.write_all(&bytes)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn to_bf16(v: &[f32]) -> Vec<bf16> {
        v.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    #[test]
    fn chat_prompt_uses_latest_message_by_default() {
        let mut session = ChatSession::new();
        session.add_user_message("first question");
        session.add_assistant_message("first answer");
        session.add_user_message("latest question");

        let prompt = session.build_prompt();
        assert!(prompt.contains("latest question"));
        assert!(!prompt.contains("first question"));
        assert!(!prompt.contains("first answer"));
        assert!(prompt.contains(PromptCondition::Reasoning.tokens()));
    }

    #[test]
    fn chat_prompt_can_include_structured_history() {
        let mut session = ChatSession::new();
        session.use_history = true;
        session.condition = PromptCondition::Direct;
        session.add_user_message("first question");
        session.add_assistant_message("first answer");
        session.add_user_message("latest question");

        let prompt = session.build_prompt();
        assert!(prompt.contains("User: first question"));
        assert!(prompt.contains("Assistant: first answer"));
        assert!(prompt.contains("User: latest question"));
        assert!(prompt.ends_with("Assistant:<|im_end|>"));
        assert!(prompt.contains(PromptCondition::Direct.tokens()));
    }

    #[test]
    fn sampler_greedy_ignores_non_finite_logits() {
        let sampler = Sampler::default();
        let logits = to_bf16(&[f32::NAN, 1.0, 3.0, 2.0]);
        assert_eq!(sampler.sample(&logits, &[], logits.len()), 2);
    }

    #[test]
    fn stable_stream_holds_incomplete_unicode() {
        let mut emitted = String::new();
        let mut chunks = Vec::new();
        emit_stable_text("caf\u{fffd}", &mut emitted, &mut |text| {
            chunks.push(text.to_string())
        });
        emit_stable_text("cafe", &mut emitted, &mut |text| {
            chunks.push(text.to_string())
        });

        assert_eq!(emitted, "cafe");
        assert_eq!(chunks, ["caf", "e"]);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_additive_inject() {
        let hrm = HrmForwardPass::new(0).unwrap();
        let a = to_bf16(&[1.0f32, 2.0, 3.0, 4.0]);
        let b = to_bf16(&[0.5f32, 1.5, 2.5, 3.5]);
        let mut out = hrm.alloc_zero_bf16(4).unwrap();

        let a_dev = hrm.htod_copy_bf16(&a).unwrap();
        let b_dev = hrm.htod_copy_bf16(&b).unwrap();

        hrm.additive_inject(&a_dev, &b_dev, &mut out, 4).unwrap();

        let out_host: Vec<bf16> = hrm.dev.sync_reclaim(out).unwrap();
        let expected = [1.5f32, 3.5, 5.5, 7.5];
        for (o, e) in out_host.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 0.01);
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_rms_norm() {
        let hrm = HrmForwardPass::new(0).unwrap();
        let inp = to_bf16(&[1.0f32, 2.0, 3.0, 4.0]);
        let mut out = hrm.alloc_zero_bf16(4).unwrap();
        let inp_dev = hrm.htod_copy_bf16(&inp).unwrap();

        hrm.rms_norm(&inp_dev, &mut out, 1, 4).unwrap();

        let out_host: Vec<bf16> = hrm.dev.sync_reclaim(out).unwrap();
        let rms = (1.0f32 + 4.0 + 9.0 + 16.0).sqrt() / 2.0;
        let expected = [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
        for (o, e) in out_host.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 0.01);
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_swiglu_ffn() {
        let hrm = HrmForwardPass::new(0).unwrap();
        let g = to_bf16(&[0.0f32; 4]);
        let u = to_bf16(&[2.0f32; 4]);
        let mut out = hrm.alloc_zero_bf16(4).unwrap();
        let g_dev = hrm.htod_copy_bf16(&g).unwrap();
        let u_dev = hrm.htod_copy_bf16(&u).unwrap();

        hrm.swiglu_ffn(&g_dev, &u_dev, &mut out, 4).unwrap();

        let out_host: Vec<bf16> = hrm.dev.sync_reclaim(out).unwrap();
        for o in out_host.iter() {
            assert!(o.to_f32().abs() < 0.01);
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_gated_attn_output() {
        let hrm = HrmForwardPass::new(0).unwrap();
        let gate = to_bf16(&[0.0f32; 4]);
        let attn = to_bf16(&[1.0f32; 4]);
        let mut out = hrm.alloc_zero_bf16(4).unwrap();
        let g_dev = hrm.htod_copy_bf16(&gate).unwrap();
        let a_dev = hrm.htod_copy_bf16(&attn).unwrap();

        hrm.gated_attn_output(&g_dev, &a_dev, &mut out, 4).unwrap();

        let out_host: Vec<bf16> = hrm.dev.sync_reclaim(out).unwrap();
        for o in out_host.iter() {
            assert!((o.to_f32() - 0.5).abs() < 0.01);
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_prefixlm_mask() {
        let hrm = HrmForwardPass::new(0).unwrap();
        let token_types = [0i32, 1, 0];
        let mut mask = hrm.alloc_zero_bf16(9).unwrap();
        let tt_dev = hrm.dev.htod_copy(token_types.to_vec()).unwrap();

        hrm.prefixlm_mask(&tt_dev, &mut mask, 1, 3).unwrap();

        let mask_host: Vec<bf16> = hrm.dev.sync_reclaim(mask).unwrap();
        assert_eq!(mask_host[0].to_f32(), 0.0);
        assert!(mask_host[1].to_f32() < -1000.0);
        assert!(mask_host[2].to_f32() < -1000.0);
        assert!(mask_host[3 + 0].to_f32() < -1000.0);
        assert_eq!(mask_host[3 + 1].to_f32(), 0.0);
        assert!(mask_host[3 + 2].to_f32() < -1000.0);
    }
}
