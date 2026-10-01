//! The model in plain f32 on the CPU, one sequence at a time: the oracle
//! the engine's logits are checked against.

use super::{Config, Weights, rope_tables};
use crate::half::f16_to_f32;

fn f(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&h| f16_to_f32(h)).collect()
}

/// y = W x for W [rows, cols].
fn matvec(w: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let cols = x.len();
    (0..rows)
        .map(|r| w[r * cols..(r + 1) * cols].iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * s * b).collect()
}

fn rope(x: &mut [f32], pos: usize, cfg: &Config, c: &[f32], s: &[f32]) {
    let h = cfg.head_dim / 2;
    for head in x.chunks_mut(cfg.head_dim) {
        for i in 0..h {
            let (cs, sn) = (c[pos * h + i], s[pos * h + i]);
            let (a, b) = (head[i], head[i + h]);
            head[i] = a * cs - b * sn;
            head[i + h] = b * cs + a * sn;
        }
    }
}

/// The logits after each token of `tokens`, run as one sequence.
pub fn forward(w: &Weights, tokens: &[i32]) -> Vec<Vec<f32>> {
    let cfg = &w.cfg;
    let (hd, nh, ng) = (cfg.head_dim, cfg.heads, cfg.kv_heads);
    let (c, s) = rope_tables(cfg);
    let embed = f(&w.embed);
    let layers: Vec<_> = w
        .layers
        .iter()
        .map(|l| {
            (
                f(&l.attn_norm),
                f(&l.wqkv),
                f(&l.wo),
                f(&l.mlp_norm),
                f(&l.wg),
                f(&l.wu),
                f(&l.wd),
            )
        })
        .collect();
    let norm = f(&w.norm);
    let head = f(w.lm_head());
    let mut kcache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
    let mut vcache: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
    let mut out = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        let mut x = embed[t as usize * cfg.dim..(t as usize + 1) * cfg.dim].to_vec();
        for (li, (n1, wqkv, wo, n2, wg, wu, wd)) in layers.iter().enumerate() {
            let h = rmsnorm(&x, n1, cfg.eps);
            let qkv = matvec(wqkv, &h, cfg.qkv());
            let mut q = qkv[..nh * hd].to_vec();
            let mut k = qkv[nh * hd..(nh + ng) * hd].to_vec();
            let v = qkv[(nh + ng) * hd..].to_vec();
            rope(&mut q, pos, cfg, &c, &s);
            rope(&mut k, pos, cfg, &c, &s);
            kcache[li].push(k);
            vcache[li].push(v);
            let mut o = vec![0f32; nh * hd];
            for hh in 0..nh {
                let g = hh / (nh / ng);
                let qh = &q[hh * hd..(hh + 1) * hd];
                let sc: Vec<f32> = kcache[li]
                    .iter()
                    .map(|k| {
                        k[g * hd..(g + 1) * hd].iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = sc.iter().map(|x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                for (p, v) in e.iter().zip(&vcache[li]) {
                    for d in 0..hd {
                        o[hh * hd + d] += p / z * v[g * hd + d];
                    }
                }
            }
            for (a, b) in x.iter_mut().zip(matvec(wo, &o, cfg.dim)) {
                *a += b;
            }
            let h = rmsnorm(&x, n2, cfg.eps);
            let g = matvec(wg, &h, cfg.ffn);
            let u = matvec(wu, &h, cfg.ffn);
            let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            for (xa, b) in x.iter_mut().zip(matvec(wd, &a, cfg.dim)) {
                *xa += b;
            }
        }
        out.push(matvec(&head, &rmsnorm(&x, &norm, cfg.eps), cfg.vocab));
    }
    out
}
