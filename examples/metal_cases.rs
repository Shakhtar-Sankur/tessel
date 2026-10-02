//! tessel's kernels for Apple GPUs, checked on a Mac: `write DIR` writes,
//! for each case, the kernel in the Metal Shading Language, its inputs and
//! launch shape (DIR/cases.json); scripts/metal_run.swift compiles and runs
//! them with Metal and saves every buffer after the launch; `check DIR`
//! compares those with the reference interpreter's outputs. `emu DIR`
//! stands in for the Mac: it runs the same kernels (as generated for
//! Metal: no tensor cores) on tessel's emulator and saves their outputs.
//!
//! usage: cargo run --release --example metal_cases -- write out
//!        swift scripts/metal_run.swift out
//!        cargo run --release --example metal_cases -- check out

use std::path::Path;
use tessel::ast::DType;
use tessel::bench::Rng;
use tessel::cuda::{METAL, Options, generate};
use tessel::interp::{self, Tensor};
use tessel::ir::{Spec, compile};
use tessel::runtime::{self, Device, from_bytes, to_bytes};

struct Case {
    file: &'static str,
    kernel: &'static str,
    meta: Vec<(&'static str, i64)>,
    warps: usize,
    args: Vec<Tensor>,
    /// Largest error allowed, relative to the largest expected value.
    tol: f32,
}

fn ints(v: Vec<usize>, shape: &[usize]) -> Tensor {
    Tensor::new(DType::I32, shape, v.into_iter().map(|x| x as f32).collect())
}

fn cases() -> Vec<Case> {
    let mut r = Rng::new(3);
    let (h, s, d) = (2, 128, 64);
    let (b, ph, pd, pages, p, mp) = (3, 4, 64, 12, 16, 4);
    let c = |file, kernel, meta: &[(&'static str, i64)], warps, args, tol| Case {
        file,
        kernel,
        meta: meta.to_vec(),
        warps,
        args,
        tol,
    };
    vec![
        c(
            "basic",
            "add",
            &[],
            4,
            vec![
                r.tensor(DType::F32, &[3000], 1.0),
                r.tensor(DType::F32, &[3000], 1.0),
                Tensor::zeros(DType::F32, &[3000]),
            ],
            1e-6,
        ),
        c(
            "basic",
            "softmax",
            &[("BR", 2), ("BC", 512)],
            4,
            vec![
                r.tensor(DType::F32, &[37, 500], 3.0),
                Tensor::zeros(DType::F32, &[37, 500]),
            ],
            1e-5,
        ),
        c(
            "basic",
            "rmsnorm",
            &[("BR", 2), ("BC", 512)],
            4,
            vec![
                r.tensor(DType::F16, &[37, 500], 1.0),
                r.tensor(DType::F16, &[500], 1.0),
                Tensor::zeros(DType::F16, &[37, 500]),
                Tensor::scalar(DType::F32, 1e-5),
            ],
            2e-3,
        ),
        c(
            "matmul",
            "matmul",
            &[("BM", 64), ("BN", 64), ("BK", 32), ("G", 2)],
            4,
            vec![
                r.tensor(DType::F16, &[200, 96], 1.0),
                r.tensor(DType::F16, &[96, 130], 1.0),
                Tensor::zeros(DType::F16, &[200, 130]),
            ],
            2e-3,
        ),
        c(
            "matmul",
            "matmul_f32",
            &[("BM", 32), ("BN", 32), ("BK", 16)],
            4,
            vec![
                r.tensor(DType::F32, &[100, 70], 1.0),
                r.tensor(DType::F32, &[70, 90], 1.0),
                Tensor::zeros(DType::F32, &[100, 90]),
            ],
            1e-5,
        ),
        c(
            "matmul",
            "linear",
            &[("BM", 32), ("BN", 64), ("BK", 32)],
            4,
            vec![
                r.tensor(DType::F16, &[50, 128], 1.0),
                r.tensor(DType::F16, &[96, 128], 0.2),
                r.tensor(DType::F16, &[96], 1.0),
                Tensor::zeros(DType::F16, &[50, 96]),
            ],
            2e-3,
        ),
        c(
            "attention",
            "flash_attention",
            &[("BM", 32), ("BN", 32)],
            4,
            vec![
                r.tensor(DType::F16, &[h, s, d], 1.0),
                r.tensor(DType::F16, &[h, s, d], 1.0),
                r.tensor(DType::F16, &[h, s, d], 1.0),
                Tensor::zeros(DType::F16, &[h, s, d]),
                Tensor::scalar(DType::F32, 0.125),
            ],
            2e-3,
        ),
        c(
            "attention",
            "paged_attention",
            &[],
            4,
            vec![
                r.tensor(DType::F16, &[b, ph, pd], 1.0),
                r.tensor(DType::F16, &[pages, p, ph, pd], 1.0),
                r.tensor(DType::F16, &[pages, p, ph, pd], 1.0),
                ints(vec![7, 2, 9, 0, 4, 11, 1, 6, 3, 10, 5, 8], &[b, mp]),
                ints(vec![5, 64, 37], &[b]),
                Tensor::zeros(DType::F16, &[b, ph, pd]),
                Tensor::scalar(DType::F32, 0.125),
            ],
            2e-3,
        ),
    ]
    .into_iter()
    .chain(llm(&mut r))
    .collect()
}

/// The LLM engine's kernels, at a small model's shapes: a step of 16
/// tokens, a prefill of two prompts packed into one step, and decoding.
fn llm(r: &mut Rng) -> Vec<Case> {
    let (dm, nh, ng, hd, f, v, npos) = (128, 4, 2, 64, 256, 300, 64);
    let (t, w, ns) = (16, (4 + 2 * 2) * 64, 64);
    let (b, np, p, mp) = (3, 12, 16, 4);
    let mm: &[(&'static str, i64)] = &[("BM", 16), ("BN", 64), ("BK", 64)];
    let c = |kernel, meta: &[(&'static str, i64)], warps, args, tol| Case {
        file: "llm",
        kernel,
        meta: meta.to_vec(),
        warps,
        args,
        tol,
    };
    let pos = || ints((0..t).map(|i| (i * 3) % npos).collect(), &[t]);
    // Two prompts of 70 and 26 tokens in one prefill step.
    let starts = ints((0..96).map(|i| if i < 70 { 0 } else { 70 }).collect(), &[96]);
    // The last two tokens are padding: slots out of range, writes dropped.
    let slots = ints(
        (0..t).map(|i| if i < t - 2 { (i * 5) % ns } else { ns }).collect(),
        &[t],
    );
    vec![
        c(
            "embed",
            &[("BD", 128)],
            4,
            vec![
                ints((0..t).map(|i| (i * 37) % v).collect(), &[t]),
                r.tensor(DType::F16, &[v, dm], 1.0),
                Tensor::zeros(DType::F32, &[t, dm]),
            ],
            0.0,
        ),
        c(
            "rmsnorm",
            &[("BC", 128)],
            4,
            vec![
                r.tensor(DType::F32, &[t, dm], 2.0),
                r.tensor(DType::F16, &[dm], 1.0),
                Tensor::zeros(DType::F16, &[t, dm]),
                Tensor::scalar(DType::F32, 1e-5),
            ],
            2e-3,
        ),
        c(
            "linear",
            mm,
            4,
            vec![
                r.tensor(DType::F16, &[t, dm], 1.0),
                r.tensor(DType::F16, &[w, dm], 0.1),
                Tensor::zeros(DType::F16, &[t, w]),
            ],
            2e-3,
        ),
        c(
            "linear_residual",
            mm,
            4,
            vec![
                r.tensor(DType::F16, &[t, f], 1.0),
                r.tensor(DType::F16, &[dm, f], 0.1),
                r.tensor(DType::F32, &[t, dm], 1.0),
            ],
            1e-4,
        ),
        c(
            "logits",
            mm,
            4,
            vec![
                r.tensor(DType::F16, &[t, dm], 1.0),
                r.tensor(DType::F16, &[v, dm], 0.1),
                Tensor::zeros(DType::F32, &[t, v]),
            ],
            1e-4,
        ),
        // Two weight tiles of 64x64 per stage would need 44 KB of the 32
        // KB of threadgroup memory Metal allows.
        c(
            "gate_up",
            &[("BM", 16), ("BN", 64), ("BK", 32)],
            4,
            vec![
                r.tensor(DType::F16, &[t, dm], 1.0),
                r.tensor(DType::F16, &[f, dm], 0.1),
                r.tensor(DType::F16, &[f, dm], 0.1),
                Tensor::zeros(DType::F16, &[t, f]),
            ],
            2e-3,
        ),
        c(
            "rope_q",
            &[],
            1,
            vec![
                r.tensor(DType::F16, &[t, w], 1.0),
                pos(),
                r.tensor(DType::F32, &[npos, hd / 2], 1.0),
                r.tensor(DType::F32, &[npos, hd / 2], 1.0),
                Tensor::zeros(DType::F16, &[t, nh, hd]),
            ],
            2e-3,
        ),
        c(
            "kv_write",
            &[],
            1,
            vec![
                r.tensor(DType::F16, &[t, w], 1.0),
                pos(),
                slots,
                r.tensor(DType::F32, &[npos, hd / 2], 1.0),
                r.tensor(DType::F32, &[npos, hd / 2], 1.0),
                Tensor::zeros(DType::F16, &[ns, ng, hd]),
                Tensor::zeros(DType::F16, &[ns, ng, hd]),
                Tensor::zeros(DType::F16, &[t, ng, hd]),
                Tensor::zeros(DType::F16, &[t, ng, hd]),
            ],
            2e-3,
        ),
        c(
            "prefill_attention",
            &[("BM", 32), ("BN", 32)],
            4,
            vec![
                r.tensor(DType::F16, &[96, nh, hd], 1.0),
                r.tensor(DType::F16, &[96, ng, hd], 1.0),
                r.tensor(DType::F16, &[96, ng, hd], 1.0),
                starts,
                Tensor::zeros(DType::F16, &[96, nh, hd]),
                Tensor::scalar(DType::F32, 0.125),
            ],
            2e-3,
        ),
        c(
            "decode_attention",
            &[],
            1,
            vec![
                r.tensor(DType::F16, &[b, nh, hd], 1.0),
                r.tensor(DType::F16, &[np, p, ng, hd], 1.0),
                r.tensor(DType::F16, &[np, p, ng, hd], 1.0),
                ints(vec![7, 2, 9, 0, 4, 11, 1, 6, 3, 10, 5, 8], &[b, mp]),
                ints(vec![5, 64, 0], &[b]),
                Tensor::zeros(DType::F16, &[b, nh, hd]),
                Tensor::scalar(DType::F32, 0.125),
            ],
            2e-3,
        ),
        c(
            "rmsnorm_rows",
            &[("BC", 128)],
            4,
            vec![
                r.tensor(DType::F32, &[t, dm], 2.0),
                ints(vec![15, 3, 0, 9], &[4]),
                r.tensor(DType::F16, &[dm], 1.0),
                Tensor::zeros(DType::F16, &[4, dm]),
                Tensor::scalar(DType::F32, 1e-5),
            ],
            2e-3,
        ),
    ]
}

fn name(c: &Case) -> String {
    format!("{}_{}", c.file, c.kernel)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (cmd, dir) = match a.as_slice() {
        [_, c, d] => (c.as_str(), Path::new(d)),
        _ => {
            eprintln!("usage: metal_cases write|emu|check DIR");
            std::process::exit(2);
        }
    };
    let mut entries = Vec::new();
    let mut failed = 0;
    for c in cases() {
        let shapes: Vec<Vec<usize>> = c
            .args
            .iter()
            .filter(|t| !t.shape.is_empty())
            .map(|t| t.shape.clone())
            .collect();
        let spec = Spec {
            shapes,
            meta: c.meta.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
        };
        let src = tessel::kernels::source(c.file).unwrap();
        let k = compile(src, c.kernel, &spec).unwrap();
        let g = generate(
            &k,
            &Options {
                warps: c.warps,
                arch: METAL,
            },
        )
        .unwrap();
        let cd = dir.join(name(&c));
        match cmd {
            "write" => {
                std::fs::create_dir_all(&cd).unwrap();
                std::fs::write(cd.join("kernel.metal"), g.metal.as_ref().unwrap()).unwrap();
                for (i, t) in c.args.iter().enumerate() {
                    std::fs::write(cd.join(format!("in{i}.bin")), to_bytes(t)).unwrap();
                }
                entries.push(format!(
                    "{{\"case\": \"{}\", \"function\": \"{}\", \"args\": {}, \"grid\": [{}, {}, {}], \"threads\": {}}}",
                    name(&c),
                    g.name,
                    c.args.len(),
                    g.grid[0],
                    g.grid[1],
                    g.grid[2],
                    g.threads
                ));
            }
            "emu" => {
                let mut got = c.args.clone();
                let o = Options {
                    warps: c.warps,
                    arch: METAL,
                };
                runtime::compile(&k, Device::Emu, &o).unwrap().run(&mut got).unwrap();
                for (i, t) in got.iter().enumerate() {
                    std::fs::write(cd.join(format!("out{i}.bin")), to_bytes(t)).unwrap();
                }
            }
            "check" => {
                let mut want = c.args.clone();
                interp::run(&k, &mut want).unwrap();
                for (i, w) in want.iter().enumerate() {
                    if w.shape.is_empty() {
                        continue;
                    }
                    let p = cd.join(format!("out{i}.bin"));
                    let Ok(bytes) = std::fs::read(&p) else {
                        println!("{}: no {} (the run did not finish)", name(&c), p.display());
                        failed += 1;
                        continue;
                    };
                    let mut got = w.clone();
                    from_bytes(&mut got, &bytes);
                    let scale = w.data.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-30);
                    let err = got
                        .data
                        .iter()
                        .zip(&w.data)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max)
                        / scale;
                    let ok = err <= c.tol
                        && got
                            .data
                            .iter()
                            .zip(&w.data)
                            .all(|(a, b)| a.is_finite() == b.is_finite());
                    failed += !ok as usize;
                    println!(
                        "{:28} arg {i}: max error {err:.2e} of the largest value  {}",
                        name(&c),
                        if ok { "ok" } else { "FAIL" }
                    );
                }
            }
            _ => panic!("unknown command {cmd}"),
        }
    }
    if cmd == "write" {
        std::fs::write(dir.join("cases.json"), format!("[{}]\n", entries.join(",\n"))).unwrap();
        println!("wrote {} cases to {}", entries.len(), dir.display());
    } else if failed > 0 {
        eprintln!("{failed} outputs differ from the interpreter");
        std::process::exit(1);
    } else {
        println!("every output matches the interpreter");
    }
}
