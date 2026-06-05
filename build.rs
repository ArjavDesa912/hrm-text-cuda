use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels");

    let out_dir = env::var("OUT_DIR").unwrap();
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

    let kernels = [
        "additive_inject",
        "rms_norm",
        "rope_embed",
        "prefixlm_mask",
        "swiglu_ffn",
        "gated_attn_output",
        "embedding_lookup",
        "split_gqkv",
        "apply_attn_gate",
        "mha",
        "mha_decode",
        "kv_cache_update",
        "broadcast_vec",
    ];

    for kernel in &kernels {
        let cu_file = format!("src/kernels/{}.cu", kernel);
        let ptx_file = format!("{}/{}.ptx", out_dir, kernel);

        let status = Command::new("nvcc")
            .args(&[
                "-ptx",
                "-O3",
                &format!("-arch={}", arch),
                "--generate-code",
                &format!("arch=compute_80,code={}", arch),
                "-o",
                &ptx_file,
                &cu_file,
                "-I",
                &out_dir,
                "--std=c++17",
            ])
            .status()
            .expect("nvcc not found; ensure CUDA toolkit is installed and on PATH");

        if !status.success() {
            panic!("nvcc failed for {}", kernel);
        }

        println!("cargo:rustc-link-search=native={}", out_dir);
    }
}
