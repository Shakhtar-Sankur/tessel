//! The tessel command line.

use std::io::Write;
use tessel::ast::DType;
use tessel::cuda::{Options, generate};
use tessel::interp::{self, Tensor};
use tessel::ir::{self, Spec};
use tessel::runtime::{self, Device};

const USAGE: &str = "tessel: a tile language for GPU and TPU kernels

usage:
  tessel ir    FILE KERNEL --shapes 128x64,64x32,128x32 [--meta BM=64,BN=64]
  tessel cuda  FILE KERNEL --shapes ... [--meta ...] [--warps 4]
  tessel pallas FILE KERNEL --shapes ... [--meta ...]
               (the kernel for TPUs, as a Python module using Pallas)
  tessel run   FILE KERNEL --shapes ... [--meta ...] [--warps 4] [--device emu|cuda]
               (random inputs; every output checked against the interpreter)
  tessel bench [--quick] [--iters 100] [--json OUT] [--tuned FILE]
               (the benchmark suite on the GPU, tuned per case; FILE keeps the
               chosen configurations, so later runs time only those)

FILE is a .tl file, or the name of a built-in one (basic, matmul, attention).";

fn die(msg: &str) -> ! {
    eprintln!("tessel: {msg}");
    std::process::exit(1)
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn source(file: &str) -> String {
    if let Some(s) = tessel::kernels::source(file) {
        return s.to_string();
    }
    std::fs::read_to_string(file).unwrap_or_else(|e| die(&format!("{file}: {e}")))
}

fn spec(args: &[String]) -> Spec {
    let shapes = flag(args, "--shapes")
        .unwrap_or_else(|| die("--shapes is required"))
        .split(',')
        .map(|s| {
            s.split('x')
                .map(|d| d.parse().unwrap_or_else(|_| die(&format!("bad shape {s}"))))
                .collect()
        })
        .collect();
    let meta = flag(args, "--meta")
        .map(|m| {
            m.split(',')
                .filter(|s| !s.is_empty())
                .map(|kv| {
                    let (k, v) = kv.split_once('=').unwrap_or_else(|| die(&format!("bad meta {kv}")));
                    (
                        k.to_string(),
                        v.parse().unwrap_or_else(|_| die(&format!("bad meta {kv}"))),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Spec { shapes, meta }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let warps: usize = flag(&args, "--warps")
        .map(|w| w.parse().unwrap_or_else(|_| die("bad --warps")))
        .unwrap_or(4);
    match cmd {
        "ir" | "cuda" | "pallas" | "run" => {
            if args.len() < 4 {
                die(USAGE);
            }
            let src = source(&args[2]);
            let k = ir::compile(&src, &args[3], &spec(&args)).unwrap_or_else(|e| die(&e));
            match cmd {
                "ir" => print!("{}", ir::dump(&k)),
                "pallas" => print!("{}", tessel::pallas::generate(&k).unwrap_or_else(|e| die(&e))),
                "cuda" => {
                    let g = generate(&k, &Options { warps, arch: 75 }).unwrap_or_else(|e| die(&e));
                    print!("{}", g.source);
                    eprintln!(
                        "grid {:?}, {} threads, {} bytes of shared memory",
                        g.grid, g.threads, g.smem
                    );
                }
                _ => {
                    let dev = match flag(&args, "--device").unwrap_or("emu") {
                        "emu" => Device::Emu,
                        "cuda" => Device::Cuda,
                        d => die(&format!("unknown device {d}")),
                    };
                    let mut rng = tessel::bench::Rng::new(1);
                    let inputs: Vec<Tensor> = k
                        .params
                        .iter()
                        .map(|p| match &p.kind {
                            ir::PKind::Array { dtype, dims } => match dtype {
                                DType::I32 | DType::Bool => Tensor::zeros(*dtype, dims),
                                _ => rng.tensor(*dtype, dims, 1.0),
                            },
                            ir::PKind::Scalar(d) => Tensor::scalar(*d, 0.125),
                        })
                        .collect();
                    let mut want = inputs.clone();
                    interp::run(&k, &mut want).unwrap_or_else(|e| die(&e));
                    let c = runtime::compile(&k, dev, &Options { warps, arch: 75 }).unwrap_or_else(|e| die(&e));
                    let mut got = inputs.clone();
                    c.run(&mut got).unwrap_or_else(|e| die(&e));
                    for (p, (g, w)) in k.params.iter().zip(got.iter().zip(&want)) {
                        if g.shape.is_empty() {
                            continue;
                        }
                        let d = g
                            .data
                            .iter()
                            .zip(&w.data)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        println!("{}: max |diff| from the interpreter {d:.3e}", p.name);
                    }
                    println!(
                        "compiled in {:.2} s, {} threads, {} bytes of shared memory",
                        c.compile_seconds, c.code.threads, c.code.smem
                    );
                }
            }
        }
        "bench" => {
            let quick = args.iter().any(|a| a == "--quick");
            let iters = flag(&args, "--iters").map(|x| x.parse().unwrap_or(100)).unwrap_or(100);
            let tuned = flag(&args, "--tuned");
            let mut file = flag(&args, "--json").map(|p| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .unwrap_or_else(|e| die(&format!("{p}: {e}")))
            });
            tessel::bench::run(quick, iters, tuned, &mut |line| {
                println!("{line}");
                if let Some(f) = file.as_mut() {
                    let _ = writeln!(f, "{line}");
                }
            })
            .unwrap_or_else(|e| die(&e));
        }
        _ => {
            println!("{USAGE}");
        }
    }
}
