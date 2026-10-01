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

/// The LLM engine's kernels at TinyLlama-1.1B's shapes (dim 2048, 32 heads,
/// 4 KV heads of 64, MLP 5632, vocab 32000), for a decode step of 16 rows
/// and a 512-token prompt, as the engine compiles them.
#[test]
fn llm_kernels_compile_at_tinyllama_shapes() {
    let nv = match runtime::nvrtc() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    let (dm, nh, ng, hd, f, v, npos) = (2048, 32, 4, 64, 5632, 32000, 2048);
    let (w, ns, np, p, mp) = ((nh + 2 * ng) * hd, 16384, 1024, 16, 128);
    for t in [16usize, 512] {
        let mm: &[(&str, i64)] = if t >= 64 {
            &[("BM", 64), ("BN", 128), ("BK", 32)]
        } else {
            &[("BM", 16), ("BN", 64), ("BK", 64)]
        };
        let gm: &[(&str, i64)] = &[("BM", 64), ("BN", 64), ("BK", 32)];
        let cases: Vec<(&str, Spec, usize)> = vec![
            ("embed", spec(&[&[t], &[v, dm], &[t, dm]], &[("BD", 1024)]), 4),
            ("rmsnorm", spec(&[&[t, dm], &[dm], &[t, dm]], &[("BC", 2048)]), 4),
            ("linear", spec(&[&[t, dm], &[w, dm], &[t, w]], mm), 4),
            (
                "rope_q",
                spec(&[&[t, w], &[t], &[npos, hd / 2], &[npos, hd / 2], &[t, nh, hd]], &[]),
                1,
            ),
            (
                "kv_write",
                spec(
                    &[
                        &[t, w],
                        &[t],
                        &[t],
                        &[npos, hd / 2],
                        &[npos, hd / 2],
                        &[ns, ng, hd],
                        &[ns, ng, hd],
                        &[t, ng, hd],
                        &[t, ng, hd],
                    ],
                    &[],
                ),
                1,
            ),
            (
                "prefill_attention",
                spec(
                    &[&[t, nh, hd], &[t, ng, hd], &[t, ng, hd], &[t], &[t, nh, hd]],
                    &[("BM", 64), ("BN", 64)],
                ),
                4,
            ),
            (
                "decode_attention",
                spec(
                    &[
                        &[t, nh, hd],
                        &[np, p, ng, hd],
                        &[np, p, ng, hd],
                        &[t, mp],
                        &[t],
                        &[t, nh, hd],
                    ],
                    &[],
                ),
                1,
            ),
            (
                "linear_residual",
                spec(&[&[t, nh * hd], &[dm, nh * hd], &[t, dm]], mm),
                4,
            ),
            (
                "gate_up",
                spec(&[&[t, dm], &[f, dm], &[f, dm], &[t, f]], if t >= 64 { gm } else { mm }),
                4,
            ),
            ("linear_residual", spec(&[&[t, f], &[dm, f], &[t, dm]], mm), 4),
            (
                "rmsnorm_rows",
                spec(&[&[t, dm], &[16], &[dm], &[16, dm]], &[("BC", 2048)]),
                4,
            ),
            (
                "logits",
                spec(&[&[16, dm], &[v, dm], &[16, v]], &[("BM", 16), ("BN", 64), ("BK", 64)]),
                4,
            ),
        ];
        for (name, sp, warps) in cases {
            let k = compile(&src("llm"), name, &sp).unwrap();
            let g = generate(&k, &Options { warps, arch: 75 }).unwrap();
            let (_, log) = nv
                .compile(&g.source, 75, true)
                .unwrap_or_else(|e| panic!("{name} T={t}: {e}"));
            let spills = log
                .lines()
                .any(|l| l.contains("bytes spill stores") && !l.contains(" 0 bytes spill stores"));
            let regs = log
                .lines()
                .find(|l| l.contains("registers"))
                .unwrap_or("")
                .trim()
                .to_string();
            eprintln!("{name} T={t}: {regs}");
            assert!(!spills, "{name} T={t} spills registers:\n{log}");
        }
    }
}
