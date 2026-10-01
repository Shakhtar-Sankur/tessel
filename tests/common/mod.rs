//! Reference implementations and inputs shared by the tests.
#![allow(dead_code)]

use tessel::ast::DType;
use tessel::interp::Tensor;

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1)
    }
    pub fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn tensor(&mut self, dtype: DType, shape: &[usize], scale: f32) -> Tensor {
        let n = shape.iter().product();
        Tensor::new(dtype, shape, (0..n).map(|_| self.uniform() * scale).collect())
    }
}

pub fn src(name: &str) -> String {
    std::fs::read_to_string(format!("{}/kernels/{name}.tl", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// Largest |a - b|, and the largest |b|.
pub fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    let s = b.iter().map(|x| x.abs()).fold(0.0, f32::max);
    (d, s)
}

pub fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    let (d, s) = max_diff(got, want);
    assert!(
        got.iter().all(|x| !x.is_nan()) && d <= tol * s.max(1.0),
        "{what}: max |diff| {d} (largest {s}, tolerance {tol})"
    );
}

pub fn softmax_ref(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut y = vec![0.0; r * c];
    for i in 0..r {
        let row = &x[i * c..(i + 1) * c];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = row.iter().map(|v| ((v - m) as f64).exp()).collect();
        let s: f64 = e.iter().sum();
        for j in 0..c {
            y[i * c + j] = (e[j] / s) as f32;
        }
    }
    y
}

pub fn rmsnorm_ref(x: &[f32], w: &[f32], r: usize, c: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0; r * c];
    for i in 0..r {
        let row = &x[i * c..(i + 1) * c];
        let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / c as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        for j in 0..c {
            y[i * c + j] = (row[j] as f64 * inv * w[j] as f64) as f32;
        }
    }
    y
}

/// C = A B with A [m, k] and B [k, n] (or B^T [n, k] when `bt`).
pub fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, bt: bool) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for l in 0..k {
                let bv = if bt { b[j * k + l] } else { b[l * n + j] };
                s += a[i * k + l] as f64 * bv as f64;
            }
            c[i * n + j] = s as f32;
        }
    }
    c
}

/// Causal attention per head; q, k, v and the result are [h, s, d].
pub fn attention_ref(q: &[f32], k: &[f32], v: &[f32], h: usize, s: usize, d: usize, scale: f32) -> Vec<f32> {
    let mut o = vec![0.0; h * s * d];
    for hh in 0..h {
        let base = hh * s * d;
        for i in 0..s {
            let sc: Vec<f64> = (0..=i)
                .map(|j| {
                    (0..d)
                        .map(|t| q[base + i * d + t] as f64 * k[base + j * d + t] as f64)
                        .sum::<f64>()
                        * scale as f64
                })
                .collect();
            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for t in 0..d {
                o[base + i * d + t] = ((0..=i).map(|j| e[j] * v[base + j * d + t] as f64).sum::<f64>() / z) as f32;
            }
        }
    }
    o
}

/// The inputs of paged decode attention, and its expected output.
pub struct Paged {
    pub args: Vec<Tensor>,
    pub want: Vec<f32>,
}

pub fn paged_case(rng: &mut Rng, b: usize, h: usize, d: usize, p: usize, lens: &[usize]) -> Paged {
    let mp = lens.iter().map(|l| l.div_ceil(p)).max().unwrap().max(1);
    let np = lens.iter().map(|l| l.div_ceil(p)).sum::<usize>() + 3;
    let q = rng.tensor(DType::F16, &[b, h, d], 1.0);
    let kc = rng.tensor(DType::F16, &[np, p, h, d], 1.0);
    let vc = rng.tensor(DType::F16, &[np, p, h, d], 1.0);
    // Pages handed out in a scrambled order.
    let mut order: Vec<usize> = (0..np).collect();
    for i in (1..np).rev() {
        let j = ((rng.uniform() + 1.0) * 0.5 * (i + 1) as f32) as usize % (i + 1);
        order.swap(i, j);
    }
    let mut table = vec![0.0f32; b * mp];
    let mut next = 0;
    for s in 0..b {
        for n in 0..lens[s].div_ceil(p) {
            table[s * mp + n] = order[next] as f32;
            next += 1;
        }
    }
    let scale = 1.0 / (d as f32).sqrt();
    let mut want = vec![0.0; b * h * d];
    for s in 0..b {
        for hh in 0..h {
            let at = |c: &Tensor, t: usize, x: usize| {
                let page = table[s * mp + t / p] as usize;
                c.data[((page * p + t % p) * h + hh) * d + x] as f64
            };
            let sc: Vec<f64> = (0..lens[s])
                .map(|t| {
                    (0..d)
                        .map(|x| q.data[(s * h + hh) * d + x] as f64 * at(&kc, t, x))
                        .sum::<f64>()
                        * scale as f64
                })
                .collect();
            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for x in 0..d {
                want[(s * h + hh) * d + x] = ((0..lens[s]).map(|t| e[t] * at(&vc, t, x)).sum::<f64>() / z) as f32;
            }
        }
    }
    let args = vec![
        q,
        kc,
        vc,
        Tensor::new(DType::I32, &[b, mp], table),
        Tensor::new(DType::I32, &[b], lens.iter().map(|&l| l as f32).collect()),
        Tensor::zeros(DType::F16, &[b, h, d]),
        Tensor::scalar(DType::F32, scale),
    ];
    Paged { args, want }
}
