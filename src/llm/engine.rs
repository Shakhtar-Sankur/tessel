//! The model on the device: weights, the paged KV cache, activation
//! buffers for the largest step, and one step of the decoder as a sequence
//! of tessel kernel launches.

use super::{Config, Weights, rope_tables};
use crate::gpu::{Arg, Buf, Gpu, GraphExec, K};
use crate::runtime::Device;

struct DevLayer {
    n1: Buf,
    wqkv: Buf,
    wo: Buf,
    n2: Buf,
    wg: Buf,
    wu: Buf,
    wd: Buf,
    kc: Buf,
    vc: Buf,
}

/// Engine limits.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Tokens per cache page.
    pub page: usize,
    /// Pages in the cache (shared by all sequences).
    pub pages: usize,
    /// Most tokens in one step (a prompt, or a batch of decodes).
    pub max_tokens: usize,
    /// Most pages one sequence holds (so its longest context is this * page).
    pub max_seq_pages: usize,
}

/// How a step's tokens attend.
pub enum Attn<'a> {
    /// One sequence's prompt from position 0: causal over the step's tokens.
    Prefill,
    /// One token per sequence, against its cache: page tables and lengths
    /// (including the token itself).
    Decode { tables: &'a [Vec<i32>], lens: &'a [i32] },
}

pub struct Engine {
    pub gpu: Gpu,
    /// Replay decode steps as CUDA graphs (one launch instead of hundreds).
    pub graphs: bool,
    graph: std::collections::HashMap<usize, GraphExec>,
    pub cfg: Config,
    pub lim: Limits,
    embed: Buf,
    head: Buf,
    norm: Buf,
    layers: Vec<DevLayer>,
    cos: Buf,
    sin: Buf,
    // Activations, sized for max_tokens rows.
    x: Buf,
    h: Buf,
    qkv: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    o: Buf,
    mlp: Buf,
    last: Buf,
    logits: Buf,
    tok: Buf,
    pos: Buf,
    slot: Buf,
    idx: Buf,
    table: Buf,
    lens: Buf,
}

fn bytes16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytesi(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Rows a step is compiled for: steps are padded to these sizes so a few
/// compilations serve every step.
pub fn padded(n: usize) -> usize {
    n.max(16).next_power_of_two()
}

/// Tile sizes for a matmul with `m` rows (`two` for gate_up's pair of
/// accumulators, which needs half the columns to stay in registers).
fn mm_meta(m: usize, two: bool) -> (Vec<(&'static str, i64)>, usize) {
    if m >= 64 {
        (vec![("BM", 64), ("BN", if two { 64 } else { 128 }), ("BK", 32)], 4)
    } else {
        (vec![("BM", 16), ("BN", 64), ("BK", 64)], 4)
    }
}

impl Engine {
    pub fn new(dev: Device, w: &Weights, lim: Limits) -> Result<Engine, String> {
        let cfg = w.cfg.clone();
        if !cfg.heads.is_multiple_of(cfg.kv_heads) || !cfg.head_dim.is_multiple_of(2) {
            return Err("heads must be a multiple of kv_heads, head_dim even".into());
        }
        let mut gpu = Gpu::new(dev)?;
        let up16 = |gpu: &mut Gpu, v: &[u16]| -> Result<Buf, String> {
            let b = gpu.alloc(v.len() * 2)?;
            gpu.write(b, 0, &bytes16(v))?;
            Ok(b)
        };
        let embed = up16(&mut gpu, &w.embed)?;
        let head = match &w.lm_head {
            Some(h) => up16(&mut gpu, h)?,
            None => embed,
        };
        let norm = up16(&mut gpu, &w.norm)?;
        let slots = lim.pages * lim.page;
        let kv_bytes = slots * cfg.kv_heads * cfg.head_dim * 2;
        let mut layers = Vec::new();
        for l in &w.layers {
            layers.push(DevLayer {
                n1: up16(&mut gpu, &l.attn_norm)?,
                wqkv: up16(&mut gpu, &l.wqkv)?,
                wo: up16(&mut gpu, &l.wo)?,
                n2: up16(&mut gpu, &l.mlp_norm)?,
                wg: up16(&mut gpu, &l.wg)?,
                wu: up16(&mut gpu, &l.wu)?,
                wd: up16(&mut gpu, &l.wd)?,
                kc: gpu.alloc(kv_bytes)?,
                vc: gpu.alloc(kv_bytes)?,
            });
        }
        let (c, s) = rope_tables(&cfg);
        let cos = gpu.alloc(c.len() * 4)?;
        gpu.write(cos, 0, &bytes32(&c))?;
        let sin = gpu.alloc(s.len() * 4)?;
        gpu.write(sin, 0, &bytes32(&s))?;
        let t = padded(lim.max_tokens);
        let hd = cfg.heads * cfg.head_dim;
        let gd = cfg.kv_heads * cfg.head_dim;
        Ok(Engine {
            x: gpu.alloc(t * cfg.dim * 4)?,
            h: gpu.alloc(t * cfg.dim * 2)?,
            qkv: gpu.alloc(t * cfg.qkv() * 2)?,
            q: gpu.alloc(t * hd * 2)?,
            k: gpu.alloc(t * gd * 2)?,
            v: gpu.alloc(t * gd * 2)?,
            o: gpu.alloc(t * hd * 2)?,
            mlp: gpu.alloc(t * cfg.ffn * 2)?,
            last: gpu.alloc(t * cfg.dim * 2)?,
            logits: gpu.alloc(t * cfg.vocab * 4)?,
            tok: gpu.alloc(t * 4)?,
            pos: gpu.alloc(t * 4)?,
            slot: gpu.alloc(t * 4)?,
            idx: gpu.alloc(t * 4)?,
            table: gpu.alloc(t * lim.max_seq_pages * 4)?,
            lens: gpu.alloc(t * 4)?,
            gpu,
            graphs: true,
            graph: std::collections::HashMap::new(),
            cfg,
            lim,
            embed,
            head,
            norm,
            layers,
            cos,
            sin,
        })
    }

    fn k(&mut self, name: &str, shapes: &[Vec<usize>], meta: &[(&str, i64)], warps: usize) -> Result<K, String> {
        self.gpu.kernel("llm", name, shapes, meta, warps)
    }

    /// Runs the decoder over `tokens` at positions `pos`, writing their keys
    /// and values to cache slots `slots`, and returns the logits of the rows
    /// in `want`.
    pub fn step(
        &mut self,
        tokens: &[i32],
        pos: &[i32],
        slots: &[i32],
        attn: Attn,
        want: &[usize],
    ) -> Result<Vec<Vec<f32>>, String> {
        let n = tokens.len();
        if n == 0 || n > self.lim.max_tokens || pos.len() != n || slots.len() != n {
            return Err(format!("a step of {n} tokens (at most {})", self.lim.max_tokens));
        }
        let t = padded(n);
        let ns = self.lim.pages * self.lim.page;
        // Inputs, padded: token 0 at position 0, writing to no slot.
        let mut tk = tokens.to_vec();
        let mut ps = pos.to_vec();
        let mut sl = slots.to_vec();
        tk.resize(t, 0);
        ps.resize(t, 0);
        sl.resize(t, ns as i32);
        self.gpu.write(self.tok, 0, &bytesi(&tk))?;
        self.gpu.write(self.pos, 0, &bytesi(&ps))?;
        self.gpu.write(self.slot, 0, &bytesi(&sl))?;
        let mp = self.lim.max_seq_pages;
        if let Attn::Decode { tables, lens } = &attn {
            let mut tb = vec![0i32; t * mp];
            let mut ln = vec![0i32; t];
            for (b, row) in tables.iter().enumerate() {
                tb[b * mp..b * mp + row.len()].copy_from_slice(row);
                ln[b] = lens[b];
            }
            self.gpu.write(self.table, 0, &bytesi(&tb))?;
            self.gpu.write(self.lens, 0, &bytesi(&ln))?;
        }
        // Rows whose logits are wanted.
        let bw = (!want.is_empty()).then(|| padded(want.len()));
        if let Some(bw) = bw {
            let mut ix: Vec<i32> = want.iter().map(|&r| r as i32).collect();
            ix.resize(bw, 0);
            self.gpu.write(self.idx, 0, &bytesi(&ix))?;
        }
        let prefill = matches!(attn, Attn::Prefill);
        if !prefill && self.graphs && self.gpu.dev == Device::Cuda && bw == Some(t) {
            // A decode step of this size: replay its recorded launches, or
            // run them and record them for the next time.
            if let Some(&g) = self.graph.get(&t) {
                self.gpu.replay(g)?;
            } else {
                self.launch_all(t, false, bw)?;
                self.gpu.sync()?;
                self.gpu.capture_begin()?;
                let r = self.launch_all(t, false, bw);
                let g = self.gpu.capture_end();
                r?;
                self.graph.insert(t, g?);
            }
        } else {
            self.launch_all(t, prefill, bw)?;
        }
        if want.is_empty() {
            return Ok(Vec::new());
        }
        let v = self.cfg.vocab;
        let all = self.gpu.read_f32(self.logits, 0, want.len() * v)?;
        Ok(all.chunks(v).map(|c| c.to_vec()).collect())
    }

    /// Launches one step's kernels for `t` rows (compiling them the first
    /// time): the decoder, then, for `bw` rows of logits, the final norm and
    /// the LM head.
    fn launch_all(&mut self, t: usize, prefill: bool, bw: Option<usize>) -> Result<(), String> {
        let cfg = self.cfg.clone();
        let (dm, nh, ng, hd) = (cfg.dim, cfg.heads, cfg.kv_heads, cfg.head_dim);
        let (w, f, v) = (cfg.qkv(), cfg.ffn, cfg.vocab);
        let ns = self.lim.pages * self.lim.page;
        let mp = self.lim.max_seq_pages;
        let bc = dm.next_power_of_two() as i64;
        let (mm, mw) = mm_meta(t, false);
        let (gm, gw) = mm_meta(t, true);
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = cfg.eps;

        let embed = self.k(
            "embed",
            &[vec![t], vec![v, dm], vec![t, dm]],
            &[("BD", bc.min(1024))],
            4,
        )?;
        let norm = self.k("rmsnorm", &[vec![t, dm], vec![dm], vec![t, dm]], &[("BC", bc)], 4)?;
        let qkv = self.k("linear", &[vec![t, dm], vec![w, dm], vec![t, w]], &mm, mw)?;
        let npos = cfg.max_pos;
        let rope = self.k(
            "rope_q",
            &[
                vec![t, w],
                vec![t],
                vec![npos, hd / 2],
                vec![npos, hd / 2],
                vec![t, nh, hd],
            ],
            &[],
            1,
        )?;
        let kvw = self.k(
            "kv_write",
            &[
                vec![t, w],
                vec![t],
                vec![t],
                vec![npos, hd / 2],
                vec![npos, hd / 2],
                vec![ns, ng, hd],
                vec![ns, ng, hd],
                vec![t, ng, hd],
                vec![t, ng, hd],
            ],
            &[],
            1,
        )?;
        let att = match prefill {
            true => self.k(
                "prefill_attention",
                &[vec![t, nh, hd], vec![t, ng, hd], vec![t, ng, hd], vec![t, nh, hd]],
                &[("BM", 64), ("BN", 64)],
                4,
            )?,
            false => {
                let p = self.lim.page;
                self.k(
                    "decode_attention",
                    &[
                        vec![t, nh, hd],
                        vec![self.lim.pages, p, ng, hd],
                        vec![self.lim.pages, p, ng, hd],
                        vec![t, mp],
                        vec![t],
                        vec![t, nh, hd],
                    ],
                    &[],
                    1,
                )?
            }
        };
        let oproj = self.k(
            "linear_residual",
            &[vec![t, nh * hd], vec![dm, nh * hd], vec![t, dm]],
            &mm,
            mw,
        )?;
        let gu = self.k("gate_up", &[vec![t, dm], vec![f, dm], vec![f, dm], vec![t, f]], &gm, gw)?;
        let down = self.k("linear_residual", &[vec![t, f], vec![dm, f], vec![t, dm]], &mm, mw)?;

        let g = &mut self.gpu;
        use Arg::{Buf as B, F32};
        g.launch(embed, &[B(self.tok), B(self.embed), B(self.x)])?;
        for l in &self.layers {
            g.launch(norm, &[B(self.x), B(l.n1), B(self.h), F32(eps)])?;
            g.launch(qkv, &[B(self.h), B(l.wqkv), B(self.qkv)])?;
            g.launch(rope, &[B(self.qkv), B(self.pos), B(self.cos), B(self.sin), B(self.q)])?;
            g.launch(
                kvw,
                &[
                    B(self.qkv),
                    B(self.pos),
                    B(self.slot),
                    B(self.cos),
                    B(self.sin),
                    B(l.kc),
                    B(l.vc),
                    B(self.k),
                    B(self.v),
                ],
            )?;
            match prefill {
                true => g.launch(att, &[B(self.q), B(self.k), B(self.v), B(self.o), F32(scale)])?,
                false => g.launch(
                    att,
                    &[
                        B(self.q),
                        B(l.kc),
                        B(l.vc),
                        B(self.table),
                        B(self.lens),
                        B(self.o),
                        F32(scale),
                    ],
                )?,
            }
            g.launch(oproj, &[B(self.o), B(l.wo), B(self.x)])?;
            g.launch(norm, &[B(self.x), B(l.n2), B(self.h), F32(eps)])?;
            g.launch(gu, &[B(self.h), B(l.wg), B(l.wu), B(self.mlp)])?;
            g.launch(down, &[B(self.mlp), B(l.wd), B(self.x)])?;
        }
        let Some(bw) = bw else { return Ok(()) };
        let (lm, lw) = mm_meta(bw, false);
        let fin = self.k(
            "rmsnorm_rows",
            &[vec![t, dm], vec![bw], vec![dm], vec![bw, dm]],
            &[("BC", bc)],
            4,
        )?;
        let head = self.k("logits", &[vec![bw, dm], vec![v, dm], vec![bw, v]], &lm, lw)?;
        let g = &mut self.gpu;
        g.launch(fin, &[B(self.x), B(self.idx), B(self.norm), B(self.last), F32(eps)])?;
        g.launch(head, &[B(self.last), B(self.head), B(self.logits)])?;
        Ok(())
    }
}
