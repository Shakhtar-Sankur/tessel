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

  tessel llm MODEL_DIR --prompts FILE [--max-new 64] [--batch 16] [--device cuda|emu]
             [--page 16] [--pages 1024] [--max-tokens 1024] [--warmup] [--json OUT]
             [--logits OUT]  (also writes the logits after the first prompt, as JSON)
             [--no-graphs]   (launch every decode step's kernels one by one, not as a CUDA graph)
               (greedy generation with the engine of tessel kernels, continuous
               batching over a paged KV cache; MODEL_DIR is a Hugging Face Llama
               checkpoint, FILE a JSON list of token-id lists)
  tessel llm-tiny DIR   (writes a small random Llama checkpoint, for trying llm)

FILE is a .tl file, or the name of a built-in one (basic, matmul, attention, llm).";

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
        "llm" => {
            if let Err(e) = llm(&args) {
                die(&e);
            }
        }
        "llm-tiny" => {
            let dir = args.get(2).unwrap_or_else(|| die(USAGE));
            let w = tessel::llm::Weights::random(&tessel::llm::Config::tiny(), 1);
            tessel::llm::safetensors::save(&w, std::path::Path::new(dir)).unwrap_or_else(|e| die(&e));
            println!("wrote a tiny random Llama to {dir}");
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

fn llm(args: &[String]) -> Result<(), String> {
    use tessel::llm::engine::{Engine, Limits};
    use tessel::llm::safetensors::{Json, load, parse_json};
    use tessel::llm::sched::{Request, generate};
    let dir = args.get(2).ok_or(USAGE)?;
    let num = |k: &str, d: usize| -> Result<usize, String> {
        flag(args, k)
            .map(|x| x.parse().map_err(|_| format!("bad {k}")))
            .unwrap_or(Ok(d))
    };
    let dev = match flag(args, "--device").unwrap_or("cuda") {
        "emu" => Device::Emu,
        "cuda" => Device::Cuda,
        d => return Err(format!("unknown device {d}")),
    };
    let path = flag(args, "--prompts").ok_or("--prompts is required")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let Json::Arr(ps) = parse_json(&text)? else {
        return Err(format!("{path}: not a JSON list"));
    };
    let max_new = num("--max-new", 64)?;
    let reqs: Vec<Request> = ps
        .iter()
        .map(|p| match p {
            Json::Arr(ids) => Ok(Request {
                prompt: ids.iter().map(|x| x.num().unwrap_or(0.0) as i32).collect(),
                max_new,
            }),
            _ => Err(format!("{path}: each prompt is a list of token ids")),
        })
        .collect::<Result<_, _>>()?;
    let t = std::time::Instant::now();
    let w = load(std::path::Path::new(dir))?;
    let load_s = t.elapsed().as_secs_f64();
    let page = num("--page", 16)?;
    let lim = Limits {
        page,
        pages: num("--pages", 1024)?,
        max_tokens: num("--max-tokens", 1024)?,
        max_seq_pages: (w.cfg.max_pos / page).max(1),
    };
    let t = std::time::Instant::now();
    let mut e = Engine::new(dev, &w, lim)?;
    e.graphs = !args.iter().any(|a| a == "--no-graphs");
    drop(w);
    let upload_s = t.elapsed().as_secs_f64();
    let batch = num("--batch", 16)?;
    if let Some(p) = flag(args, "--logits") {
        use tessel::llm::engine::Attn;
        let pr = &reqs[0].prompt;
        let n = pr.len();
        let pos: Vec<i32> = (0..n as i32).collect();
        let l = e.step(pr, &pos, &pos, Attn::Prefill, &[n - 1])?;
        let v: Vec<String> = l[0].iter().map(|x| format!("{x:e}")).collect();
        std::fs::write(p, format!("[{}]\n", v.join(", "))).map_err(|e| format!("{p}: {e}"))?;
    }
    let mut warm_s = 0.0;
    if args.iter().any(|a| a == "--warmup") {
        // The same work once, so the timed run finds every kernel compiled.
        let t = std::time::Instant::now();
        generate(&mut e, &reqs, batch)?;
        warm_s = t.elapsed().as_secs_f64();
    }
    let (outs, st) = generate(&mut e, &reqs, batch)?;
    let ids: Vec<String> = outs
        .iter()
        .map(|o| {
            format!(
                "[{}]",
                o.tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ")
            )
        })
        .collect();
    let ttft: Vec<String> = outs.iter().map(|o| format!("{:.4}", o.first_token_s)).collect();
    let line = format!(
        "{{\"engine\": \"tessel\", \"device\": \"{:?}\", \"requests\": {}, \"batch\": {batch}, \"max_new\": {max_new}, \"prompt_tokens\": {}, \"generated\": {}, \"seconds\": {:.4}, \"tokens_per_s\": {:.2}, \"decode_steps\": {}, \"decode_seconds\": {:.4}, \"decode_tokens_per_s\": {:.2}, \"largest_batch\": {}, \"graphs\": {}, \"load_s\": {load_s:.2}, \"upload_s\": {upload_s:.2}, \"warmup_s\": {warm_s:.2}, \"first_token_s\": [{}], \"outputs\": [{}]}}",
        dev,
        reqs.len(),
        st.prompt_tokens,
        st.generated,
        st.seconds,
        st.generated as f64 / st.seconds,
        st.decode_steps,
        st.decode_seconds,
        st.decode_tokens as f64 / st.decode_seconds.max(1e-9),
        st.largest_batch,
        e.graphs,
        ttft.join(", "),
        ids.join(", ")
    );
    println!("{line}");
    if let Some(p) = flag(args, "--json") {
        std::fs::write(p, format!("{line}\n")).map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}
