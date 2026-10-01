//! The benchmark suite: each kernel at the sizes that matter on a GPU,
//! tuned over tile shapes and warp counts on the device, and checked
//! against a CPU reference (sampled for the large ones).

use crate::ast::DType;
use crate::cuda::Options;
use crate::interp::Tensor;
use crate::ir::{Spec, compile};
use crate::kernels;
use crate::runtime::{self, Device};
use std::collections::HashMap;

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1)
    }
    pub fn bits(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn uniform(&mut self) -> f32 {
        ((self.bits() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.bits() % n as u64) as usize
    }
    pub fn tensor(&mut self, dtype: DType, shape: &[usize], scale: f32) -> Tensor {
        let n = shape.iter().product();
        Tensor::new(dtype, shape, (0..n).map(|_| self.uniform() * scale).collect())
    }
}

type Verify = Box<dyn Fn(&[Tensor]) -> f32>;

pub struct Case {
    pub kind: &'static str,
    pub file: &'static str,
    pub kernel: &'static str,
    pub label: String,
    pub args: Vec<Tensor>,
    /// Floating-point operations and bytes moved per run, for TFLOPS and GB/s.
    pub flops: f64,
    pub bytes: f64,
    pub configs: Vec<(Vec<(&'static str, i64)>, usize)>,
    pub verify: Verify,
}

fn rel(err: f64, scale: f64) -> f32 {
    (err / scale.max(1e-6)) as f32
}

/// C = A B, f16 in and out; 256 sampled entries checked.
fn gemm(r: &mut Rng, m: usize, k: usize, n: usize, quick: bool) -> Case {
    let args = vec![
        r.tensor(DType::F16, &[m, k], 1.0),
        r.tensor(DType::F16, &[k, n], 1.0),
        Tensor::zeros(DType::F16, &[m, n]),
    ];
    let mut configs: Vec<(Vec<(&str, i64)>, usize)> = vec![
        (vec![("BM", 128), ("BN", 128), ("BK", 32)], 4),
        (vec![("BM", 128), ("BN", 128), ("BK", 32)], 8),
        (vec![("BM", 128), ("BN", 64), ("BK", 32)], 4),
        (vec![("BM", 64), ("BN", 128), ("BK", 32)], 4),
        (vec![("BM", 64), ("BN", 64), ("BK", 32)], 4),
        (vec![("BM", 128), ("BN", 256), ("BK", 32)], 8),
        (vec![("BM", 256), ("BN", 128), ("BK", 32)], 8),
        (vec![("BM", 64), ("BN", 64), ("BK", 64)], 4),
    ];
    if quick {
        configs.truncate(2);
    }
    let seed = r.bits();
    Case {
        kind: "gemm",
        file: "matmul",
        kernel: "matmul",
        label: format!("{m}x{k}x{n}"),
        args,
        flops: 2.0 * (m * k * n) as f64,
        bytes: 2.0 * (m * k + k * n + m * n) as f64,
        configs,
        verify: Box::new(move |a: &[Tensor]| {
            let mut r = Rng::new(seed);
            let (mut err, mut scale) = (0f64, 0f64);
            for _ in 0..256 {
                let (i, j) = (r.below(m), r.below(n));
                let want: f64 = (0..k)
                    .map(|l| a[0].data[i * k + l] as f64 * a[1].data[l * n + j] as f64)
                    .sum();
                err = err.max((a[2].data[i * n + j] as f64 - want).abs());
                scale = scale.max(want.abs());
            }
            rel(err, scale)
        }),
    }
}

/// Causal FlashAttention over [H, S, D]; 4 sampled query rows checked.
fn attention(r: &mut Rng, h: usize, s: usize, d: usize, quick: bool) -> Case {
    let scale = 1.0 / (d as f32).sqrt();
    let args = vec![
        r.tensor(DType::F16, &[h, s, d], 1.0),
        r.tensor(DType::F16, &[h, s, d], 1.0),
        r.tensor(DType::F16, &[h, s, d], 1.0),
        Tensor::zeros(DType::F16, &[h, s, d]),
        Tensor::scalar(DType::F32, scale),
    ];
    let mut configs: Vec<(Vec<(&str, i64)>, usize)> = vec![
        (vec![("BM", 64), ("BN", 64)], 4),
        (vec![("BM", 128), ("BN", 64)], 8),
        (vec![("BM", 64), ("BN", 32)], 4),
        (vec![("BM", 128), ("BN", 32)], 8),
        (vec![("BM", 32), ("BN", 64)], 2),
        (vec![("BM", 128), ("BN", 64)], 4),
        (vec![("BM", 128), ("BN", 128)], 8),
    ];
    if quick {
        configs.truncate(2);
    }
    let seed = r.bits();
    Case {
        kind: "attention",
        file: "attention",
        kernel: "flash_attention",
        label: format!("H{h} S{s} D{d} causal"),
        args,
        // QK^T and PV, half of each under the causal mask.
        flops: 2.0 * 2.0 * (h * s * s * d) as f64 / 2.0,
        bytes: 2.0 * 4.0 * (h * s * d) as f64,
        configs,
        verify: Box::new(move |a: &[Tensor]| {
            let mut r = Rng::new(seed);
            let (mut err, mut sc) = (0f64, 0f64);
            for _ in 0..4 {
                let (hh, i) = (r.below(h), r.below(s));
                let base = hh * s * d;
                let q = &a[0].data[base + i * d..base + (i + 1) * d];
                let scores: Vec<f64> = (0..=i)
                    .map(|j| {
                        (0..d)
                            .map(|t| q[t] as f64 * a[1].data[base + j * d + t] as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for t in 0..d {
                    let want = (0..=i).map(|j| e[j] * a[2].data[base + j * d + t] as f64).sum::<f64>() / z;
                    err = err.max((a[3].data[base + i * d + t] as f64 - want).abs());
                    sc = sc.max(want.abs());
                }
            }
            rel(err, sc)
        }),
    }
}

fn softmax(r: &mut Rng, rows: usize, cols: usize) -> Case {
    let args = vec![
        r.tensor(DType::F32, &[rows, cols], 4.0),
        Tensor::zeros(DType::F32, &[rows, cols]),
    ];
    let configs = vec![
        (vec![("BR", 1), ("BC", cols as i64)], 4),
        (vec![("BR", 2), ("BC", cols as i64)], 4),
        (vec![("BR", 4), ("BC", cols as i64)], 8),
    ];
    Case {
        kind: "softmax",
        file: "basic",
        kernel: "softmax",
        label: format!("{rows}x{cols} f32"),
        args,
        flops: 0.0,
        bytes: 8.0 * (rows * cols) as f64,
        configs,
        verify: Box::new(move |a: &[Tensor]| {
            let mut err = 0f64;
            for i in (0..rows).step_by(97) {
                let x = &a[0].data[i * cols..(i + 1) * cols];
                let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let z: f64 = x.iter().map(|v| (*v as f64 - m).exp()).sum();
                let y = &a[1].data[i * cols..(i + 1) * cols];
                for (yj, xj) in y.iter().zip(x) {
                    err = err.max((*yj as f64 - (*xj as f64 - m).exp() / z).abs());
                }
            }
            err as f32
        }),
    }
}

fn rmsnorm(r: &mut Rng, rows: usize, cols: usize) -> Case {
    let args = vec![
        r.tensor(DType::F16, &[rows, cols], 2.0),
        r.tensor(DType::F16, &[cols], 1.0),
        Tensor::zeros(DType::F16, &[rows, cols]),
        Tensor::scalar(DType::F32, 1e-5),
    ];
    let configs = vec![
        (vec![("BR", 1), ("BC", cols as i64)], 4),
        (vec![("BR", 2), ("BC", cols as i64)], 4),
        (vec![("BR", 4), ("BC", cols as i64)], 8),
    ];
    Case {
        kind: "rmsnorm",
        file: "basic",
        kernel: "rmsnorm",
        label: format!("{rows}x{cols} f16"),
        args,
        flops: 0.0,
        bytes: 4.0 * (rows * cols) as f64,
        configs,
        verify: Box::new(move |a: &[Tensor]| {
            let (mut err, mut sc) = (0f64, 0f64);
            for i in (0..rows).step_by(97) {
                let x = &a[0].data[i * cols..(i + 1) * cols];
                let ms = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / cols as f64;
                let inv = 1.0 / (ms + 1e-5).sqrt();
                let y = &a[2].data[i * cols..(i + 1) * cols];
                for ((xj, wj), yj) in x.iter().zip(&a[1].data).zip(y) {
                    let want = *xj as f64 * inv * *wj as f64;
                    err = err.max((*yj as f64 - want).abs());
                    sc = sc.max(want.abs());
                }
            }
            rel(err, sc)
        }),
    }
}

/// The suite: `quick` for a smoke run.
pub fn suite(quick: bool) -> Vec<Case> {
    let mut r = Rng::new(7);
    let mut v = Vec::new();
    let sizes: &[usize] = if quick { &[1024] } else { &[1024, 2048, 4096] };
    for &n in sizes {
        v.push(gemm(&mut r, n, n, n, quick));
    }
    let seqs: &[usize] = if quick { &[1024] } else { &[512, 1024, 2048, 4096] };
    for &s in seqs {
        v.push(attention(&mut r, 32, s, 64, quick));
    }
    v.push(softmax(&mut r, 4096, 4096));
    v.push(rmsnorm(&mut r, 4096, 4096));
    v
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn config_name(meta: &[(&str, i64)], warps: usize) -> String {
    format!("{meta:?} warps {warps}")
}

/// Keeps the GPU busy for about a second, so timings start at the clock
/// the GPU sustains rather than its idle boost.
fn warm_up() -> Result<(), String> {
    let n = 2048;
    let spec = Spec {
        shapes: vec![vec![n, n]; 3],
        meta: vec![],
    };
    let k = compile(kernels::source("matmul").unwrap(), "matmul", &spec)?;
    let c = runtime::compile(&k, Device::Cuda, &Options::default())?;
    let mut r = Rng::new(1);
    let args = vec![
        r.tensor(DType::F16, &[n, n], 1.0),
        r.tensor(DType::F16, &[n, n], 1.0),
        Tensor::zeros(DType::F16, &[n, n]),
    ];
    let d = c.upload(&args)?;
    let t = std::time::Instant::now();
    while t.elapsed().as_secs_f64() < 1.5 {
        for _ in 0..20 {
            c.launch(&d)?;
        }
        runtime::cuda()?.sync()?;
    }
    Ok(())
}

/// Tunes (or takes the configuration tuned before, from `tuned`) and times
/// every case on the GPU; one JSON line per case.
pub fn run(quick: bool, iters: usize, tuned: Option<&str>, out: &mut dyn FnMut(&str)) -> Result<(), String> {
    let cuda = runtime::cuda()?;
    let mut cache: HashMap<String, String> = HashMap::new();
    if let Some(p) = tuned
        && let Ok(text) = std::fs::read_to_string(p)
    {
        for l in text.lines() {
            if let Some((k, v)) = l.split_once('\t') {
                cache.insert(k.to_string(), v.to_string());
            }
        }
    }
    warm_up()?;
    for case in suite(quick) {
        let src = kernels::source(case.file).unwrap();
        let shapes: Vec<Vec<usize>> = case
            .args
            .iter()
            .filter(|t| !t.shape.is_empty())
            .map(|t| t.shape.clone())
            .collect();
        let key = format!("{}|{}|{}", cuda.name, case.kind, case.label);
        let chosen = cache.get(&key).cloned();
        let candidates: Vec<&(Vec<(&str, i64)>, usize)> = case
            .configs
            .iter()
            .filter(|(m, w)| chosen.as_ref().is_none_or(|c| *c == config_name(m, *w)))
            .collect();
        let mut tried = Vec::new();
        let mut best: Option<(f64, runtime::Compiled, String)> = None;
        let mut dev_args = None;
        for (meta, warps) in candidates {
            let name = config_name(meta, *warps);
            let spec = Spec {
                shapes: shapes.clone(),
                meta: meta.iter().map(|(n, x)| (n.to_string(), *x)).collect(),
            };
            let c = compile(src, case.kernel, &spec).and_then(|k| {
                runtime::compile(
                    &k,
                    Device::Cuda,
                    &Options {
                        warps: *warps,
                        arch: 75,
                    },
                )
            });
            let c = match c {
                Ok(c) => c,
                Err(e) => {
                    tried.push(format!(
                        "{{\"config\": {}, \"error\": {}}}",
                        json_str(&name),
                        json_str(e.lines().next().unwrap_or(""))
                    ));
                    continue;
                }
            };
            if dev_args.is_none() {
                dev_args = Some(c.upload(&case.args)?);
            }
            // Chosen by the fastest of its launches: on a GPU whose clocks
            // throttle, a slower launch says more about the clocks than the
            // kernel.
            let (med, min) = c.time(dev_args.as_ref().unwrap(), 30)?;
            tried.push(format!(
                "{{\"config\": {}, \"median_ms\": {med:.4}, \"min_ms\": {min:.4}}}",
                json_str(&name)
            ));
            if best.as_ref().is_none_or(|b| min < b.0) {
                best = Some((min, c, name));
            }
        }
        let Some((_, c, name)) = best else {
            out(&format!(
                "{{\"kind\": {}, \"label\": {}, \"engine\": \"tessel\", \"error\": \"no configuration compiled\", \"tried\": [{}]}}",
                json_str(case.kind),
                json_str(&case.label),
                tried.join(", ")
            ));
            continue;
        };
        if chosen.is_none()
            && let Some(p) = tuned
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(f, "{key}\t{name}");
            }
        }
        // The chosen configuration, timed again with more launches.
        let (med, min) = c.time(dev_args.as_ref().unwrap(), iters)?;
        let d = c.upload(&case.args)?;
        c.launch(&d)?;
        let mut got = case.args.clone();
        c.download(&d, &mut got)?;
        let err = (case.verify)(&got);
        let tflops = if case.flops > 0.0 {
            case.flops / (med * 1e-3) / 1e12
        } else {
            0.0
        };
        let gbps = case.bytes / (med * 1e-3) / 1e9;
        out(&format!(
            "{{\"kind\": {}, \"label\": {}, \"engine\": \"tessel\", \"device\": {}, \"median_ms\": {med:.4}, \"min_ms\": {min:.4}, \"tflops\": {tflops:.3}, \"gbps\": {gbps:.1}, \"rel_err\": {err:.3e}, \"config\": {}, \"tried\": [{}]}}",
            json_str(case.kind),
            json_str(&case.label),
            json_str(&cuda.name),
            json_str(&name),
            tried.join(", ")
        ));
    }
    Ok(())
}
