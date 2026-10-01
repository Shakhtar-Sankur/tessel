//! Compiles every matmul tuning candidate of the LLM engine at
//! TinyLlama-1.1B's shapes with NVRTC for sm_75 and reports registers and
//! spills: a check, without a GPU, that the tuner's choices compile cleanly.
//!
//! usage: TESSEL_NVRTC=... cargo run --release --example spills

use tessel::cuda::{Options, generate};
use tessel::ir::{Spec, compile};
use tessel::llm::engine::mm_candidates;

fn main() {
    let nv = tessel::runtime::nvrtc().expect("NVRTC");
    let src = tessel::kernels::source("llm").unwrap();
    let (dm, w, f, v) = (2048usize, 2560usize, 5632usize, 32000usize);
    let mut bad = 0;
    for t in [16usize, 32] {
        let cases: Vec<(&str, Vec<Vec<usize>>, bool)> = vec![
            ("linear", vec![vec![t, dm], vec![w, dm], vec![t, w]], false),
            ("linear_residual", vec![vec![t, f], vec![dm, f], vec![t, dm]], false),
            ("gate_up", vec![vec![t, dm], vec![f, dm], vec![f, dm], vec![t, f]], true),
            ("logits", vec![vec![t, dm], vec![v, dm], vec![t, v]], false),
        ];
        for (name, shapes, two) in cases {
            for (meta, warps) in mm_candidates(t, two) {
                let spec = Spec {
                    shapes: shapes.clone(),
                    meta: meta.iter().map(|(a, b)| (a.to_string(), *b)).collect(),
                };
                let k = compile(src, name, &spec).unwrap();
                let g = generate(&k, &Options { warps, arch: 75 }).unwrap();
                let (_, log) = nv.compile(&g.source, 75, true).unwrap();
                let spill = log
                    .lines()
                    .any(|l| l.contains("bytes spill stores") && !l.contains(" 0 bytes spill stores"));
                let regs = log
                    .lines()
                    .find(|l| l.contains("registers"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
                bad += spill as usize;
                println!(
                    "t={t} {name} {meta:?} w{warps}: {}{}",
                    regs.trim_start_matches("ptxas info    : "),
                    if spill { "  SPILLS" } else { "" }
                );
            }
        }
    }
    println!("{bad} configurations spill");
}
