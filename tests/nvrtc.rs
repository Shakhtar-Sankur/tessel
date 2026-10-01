//! Every kernel's generated CUDA compiles with NVRTC for the T4 (sm_75)
//! and the A100 (sm_80), in several configurations, without spilling
//! registers in the default ones. Skipped where NVRTC is missing.

mod common;

use common::*;
use tessel::cuda::{Options, generate};
use tessel::ir::{Spec, compile};
use tessel::runtime;

fn spec(shapes: &[&[usize]], meta: &[(&str, i64)]) -> Spec {
    Spec {
        shapes: shapes.iter().map(|s| s.to_vec()).collect(),
        meta: meta.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
    }
}

#[test]
fn kernels_compile_for_t4_and_a100() {
    let nv = match runtime::nvrtc() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    let (h, s, d) = (8, 1024, 64);
    let cases: Vec<(&str, &str, Spec, usize, bool)> = vec![
        (
            "basic",
            "add",
            spec(&[&[1 << 20], &[1 << 20], &[1 << 20]], &[]),
            4,
            true,
        ),
        ("basic", "softmax", spec(&[&[4096, 1024], &[4096, 1024]], &[]), 4, true),
        (
            "basic",
            "rmsnorm",
            spec(&[&[4096, 1024], &[1024], &[4096, 1024]], &[]),
            4,
            true,
        ),
        (
            "matmul",
            "matmul",
            spec(&[&[2048, 2048], &[2048, 2048], &[2048, 2048]], &[]),
            4,
            true,
        ),
        (
            "matmul",
            "matmul",
            spec(
                &[&[2048, 2048], &[2048, 2048], &[2048, 2048]],
                &[("BM", 64), ("BN", 64)],
            ),
            4,
            true,
        ),
        (
            "matmul",
            "matmul_f32",
            spec(&[&[1024, 1024], &[1024, 1024], &[1024, 1024]], &[]),
            4,
            true,
        ),
        (
            "matmul",
            "linear",
            spec(&[&[512, 1024], &[4096, 1024], &[4096], &[512, 4096]], &[]),
            4,
            true,
        ),
        (
            "attention",
            "flash_attention",
            spec(&[&[h, s, d], &[h, s, d], &[h, s, d], &[h, s, d]], &[]),
            4,
            true,
        ),
        (
            "attention",
            "paged_attention",
            spec(
                &[
                    &[16, h, d],
                    &[256, 16, h, d],
                    &[256, 16, h, d],
                    &[16, 64],
                    &[16],
                    &[16, h, d],
                ],
                &[],
            ),
            4,
            true,
        ),
    ];
    for (file, name, sp, warps, strict) in cases {
        let k = compile(&src(file), name, &sp).unwrap();
        let g = generate(&k, &Options { warps, arch: 75 }).unwrap();
        for arch in [75, 80] {
            let (_, log) = nv
                .compile(&g.source, arch, true)
                .unwrap_or_else(|e| panic!("{name} sm_{arch}: {e}"));
            let spills = log
                .lines()
                .any(|l| l.contains("bytes spill stores") && !l.contains(" 0 bytes spill stores"));
            let regs = log
                .lines()
                .find(|l| l.contains("registers"))
                .unwrap_or("")
                .trim()
                .to_string();
            eprintln!("{name} sm_{arch}: {regs}");
            assert!(!(strict && spills), "{name} sm_{arch} spills registers:\n{log}");
        }
    }
}
