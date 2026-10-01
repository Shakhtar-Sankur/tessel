//! An LLM inference engine whose GPU work is all tessel kernels
//! (kernels/llm.tl): a Llama-family decoder with a paged KV cache and
//! continuous batching.

pub mod engine;
pub mod reference;
pub mod safetensors;
pub mod sched;

use crate::bench::Rng;
use crate::half::{f16_to_f32, f32_to_f16};

/// The model's shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub max_pos: usize,
    pub bos: i32,
    pub eos: i32,
}

impl Config {
    /// Width of the fused QKV projection's output.
    pub fn qkv(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    /// A small model for tests.
    pub fn tiny() -> Config {
        Config {
            dim: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            ffn: 128,
            vocab: 256,
            eps: 1e-5,
            rope_theta: 10000.0,
            max_pos: 256,
            bos: 1,
            eos: 2,
        }
    }
}

/// One decoder layer's weights, as f16 bits, matrices [out, in].
#[derive(Clone)]
pub struct Layer {
    pub attn_norm: Vec<u16>,
    /// q, k and v projections stacked: [(H + 2G) * D, dim].
    pub wqkv: Vec<u16>,
    pub wo: Vec<u16>,
    pub mlp_norm: Vec<u16>,
    pub wg: Vec<u16>,
    pub wu: Vec<u16>,
    pub wd: Vec<u16>,
}

#[derive(Clone)]
pub struct Weights {
    pub cfg: Config,
    pub embed: Vec<u16>,
    pub layers: Vec<Layer>,
    pub norm: Vec<u16>,
    /// None when tied to the embedding.
    pub lm_head: Option<Vec<u16>>,
}

impl Weights {
    /// Random weights scaled as initialization would, for tests.
    pub fn random(cfg: &Config, seed: u64) -> Weights {
        let mut r = Rng::new(seed);
        let mut m = |rows: usize, cols: usize| -> Vec<u16> {
            let s = 1.0 / (cols as f32).sqrt();
            (0..rows * cols).map(|_| f32_to_f16(r.uniform() * s)).collect()
        };
        let layers = (0..cfg.layers)
            .map(|_| Layer {
                attn_norm: vec![f32_to_f16(1.0); cfg.dim],
                wqkv: m(cfg.qkv(), cfg.dim),
                wo: m(cfg.dim, cfg.heads * cfg.head_dim),
                mlp_norm: vec![f32_to_f16(1.0); cfg.dim],
                wg: m(cfg.ffn, cfg.dim),
                wu: m(cfg.ffn, cfg.dim),
                wd: m(cfg.dim, cfg.ffn),
            })
            .collect();
        let embed = m(cfg.vocab, cfg.dim)
            .iter()
            .map(|&h| f32_to_f16(f16_to_f32(h) * 8.0))
            .collect();
        Weights {
            cfg: cfg.clone(),
            embed,
            layers,
            norm: vec![f32_to_f16(1.0); cfg.dim],
            lm_head: Some(m(cfg.vocab, cfg.dim)),
        }
    }

    pub fn lm_head(&self) -> &[u16] {
        self.lm_head.as_deref().unwrap_or(&self.embed)
    }
}

/// cos and sin of every position's rotation angles: [max_pos, D/2] each.
pub fn rope_tables(cfg: &Config) -> (Vec<f32>, Vec<f32>) {
    let h = cfg.head_dim / 2;
    let mut c = Vec::with_capacity(cfg.max_pos * h);
    let mut s = Vec::with_capacity(cfg.max_pos * h);
    for p in 0..cfg.max_pos {
        for i in 0..h {
            let inv = (cfg.rope_theta as f64).powf(-((2 * i) as f64) / cfg.head_dim as f64);
            let a = p as f64 * inv;
            c.push(a.cos() as f32);
            s.push(a.sin() as f32);
        }
    }
    (c, s)
}

/// The index of the largest value (the first, on ties).
pub fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}
