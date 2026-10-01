//! The TPU backend refuses block accesses it cannot prove inside their
//! arrays, since a DMA past an array's end is an error on a TPU. (Running
//! the generated kernels needs JAX: scripts/pallas_check.py.)

use tessel::ir::{Spec, compile};
use tessel::kernels::source;
use tessel::pallas::generate;

fn emit(file: &str, name: &str, shapes: &[&[usize]], meta: &[(&str, i64)]) -> Result<String, String> {
    let spec = Spec {
        shapes: shapes.iter().map(|s| s.to_vec()).collect(),
        meta: meta.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
    };
    generate(&compile(source(file).unwrap(), name, &spec).unwrap())
}

#[test]
fn proves_blocks_in_bounds() {
    assert!(emit("matmul", "matmul", &[&[512, 256], &[256, 384], &[512, 384]], &[]).is_ok());
    let fa = emit(
        "attention",
        "flash_attention",
        &[&[2usize, 256, 64] as &[usize]; 4],
        &[],
    )
    .unwrap();
    assert!(fa.contains("jax.lax.fori_loop") && fa.contains("preferred_element_type=jnp.float32"));
    // Page numbers come from the block table, so they are clamped and the
    // loaded blocks masked.
    let pa = emit(
        "attention",
        "paged_attention",
        &[
            &[3, 4, 64],
            &[12, 16, 4, 64],
            &[12, 16, 4, 64],
            &[3, 4],
            &[3],
            &[3, 4, 64],
        ],
        &[],
    )
    .unwrap();
    assert!(pa.contains("jnp.clip("));
}

#[test]
fn refuses_partial_blocks() {
    let e = emit("matmul", "matmul", &[&[500, 256], &[256, 384], &[500, 384]], &[]).unwrap_err();
    assert!(e.contains("provably inside"), "{e}");
    let e = emit("basic", "softmax", &[&[64, 1000], &[64, 1000]], &[]).unwrap_err();
    assert!(e.contains("provably inside"), "{e}");
}
