//! CUDA code generation from the tile IR.
//!
//! Every tile value gets a linear layout (`layout.rs`): which thread holds
//! which element in which register. Layouts start at *anchors*: a load
//! gets a coalescing layout, a matmul's result the tensor-core accumulator
//! fragment layout, a reduction the slice of its input's layout, and the
//! rest follows (an elementwise operation takes its most constrained
//! operand's layout). Index arithmetic (`arange`, splats, broadcasts and
//! elementwise operations on them) gets no layout at all: it is recomputed
//! directly in whatever layout its user needs, so a causal mask is built
//! in the accumulator's registers instead of being moved there. When a
//! value is needed in a layout where its threads do not already hold the
//! elements, it goes through shared memory.
//!
//! The generated code is straight-line per register (each tile is a small
//! array indexed by constants, which the compiler keeps in registers),
//! with loops only where the kernel has them and inside matmuls.

use crate::ast::DType;
use crate::ir::{Bin, Inst, Ix, Kernel, Op, PKind, Red, Ty, Un, V};
use crate::layout::Layout;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

pub const PRELUDE: &str = include_str!("prelude.cuh");
/// The same helpers in the Metal Shading Language.
pub const METAL_PRELUDE: &str = include_str!("prelude.metal");

#[derive(Clone, Debug)]
pub struct Options {
    pub warps: usize,
    /// sm_75 (Turing, the T4) or sm_80 and newer; or `METAL` (0) for
    /// Apple GPUs, where matmuls run on the SIMT path.
    pub arch: u32,
}

/// `Options::arch` for Metal.
pub const METAL: u32 = 0;

impl Default for Options {
    fn default() -> Self {
        Options { warps: 4, arch: 75 }
    }
}

#[derive(Clone, Debug)]
pub struct Generated {
    /// For `METAL`: the kernel in the Metal Shading Language (`source` is
    /// then the same kernel for the emulator).
    pub metal: Option<String>,
    pub name: String,
    pub source: String,
    pub threads: usize,
    pub smem: usize,
    pub grid: [usize; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Lay {
    Unset,
    Scalar,
    Lazy,
    Pending,
    L(usize),
}

struct G<'a> {
    k: &'a Kernel,
    o: &'a Options,
    layouts: Vec<Layout>,
    index: HashMap<Layout, usize>,
    prio: Vec<u8>,
    lay: Vec<Lay>,
    def: Vec<Option<&'a Inst>>,
    div: Vec<u64>,
    /// Ranges of integer scalars, where known.
    rng: Vec<Option<(i64, i64)>>,
    out: String,
    ind: usize,
    scopes: Vec<HashMap<(V, usize), String>>,
    scal: HashMap<V, String>,
    n: usize,
    scratch: usize,
    dot_smem: usize,
    used_tb: BTreeSet<usize>,
    /// Layouts that are tensor-core accumulators.
    mma: std::collections::HashSet<usize>,
    /// f16 loads kept as packed halves (they only feed matmuls), with
    /// their vector width.
    raw: HashMap<V, usize>,
    /// Loads already issued by a loop's software pipeline.
    prefetched: std::collections::HashSet<V>,
    /// Parity variables of the enclosing loops (double-buffered staging).
    loops: Vec<String>,
    /// Matmul operands staged once before the loop that uses them:
    /// (value, is A) -> (shared-memory pointer, row length, transposed).
    prestaged: HashMap<(V, bool), (String, usize, bool)>,
    /// Some matmul takes a computed tile as A (attention's P @ V): keep
    /// warps along rows so it can be fed from registers.
    rowwise: bool,
}

/// Cache key of a value's packed-halves form.
const RAW: usize = usize::MAX;

const BIG: u64 = 1 << 20;

/// A slice dimension of a block access: (array stride, start expression,
/// array dimension, start value).
type SliceDim = (usize, String, usize, V);
/// A block access's base offset, point-index condition and slice dimensions.
type Access = (String, String, Vec<SliceDim>);

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn cty(d: DType) -> &'static str {
    match d {
        DType::F32 | DType::F16 => "float",
        DType::I32 => "int",
        DType::Bool => "bool",
    }
}

fn lit(x: f64, d: DType) -> String {
    match d {
        DType::I32 => format!("{}", x as i64),
        DType::Bool => (if x != 0.0 { "true" } else { "false" }).into(),
        _ => {
            if x.is_nan() {
                "KNAN".into()
            } else if x.is_infinite() {
                (if x > 0.0 { "KINF" } else { "(-KINF)" }).into()
            } else {
                let v = x as f32;
                if v == v.trunc() && v.abs() < 1e7 {
                    format!("{v:.1}f")
                } else {
                    format!("{v:e}f")
                }
            }
        }
    }
}

fn log2(n: usize) -> u32 {
    n.trailing_zeros()
}

/// The offset (in halves) of row `r`, column `c` of a staged matmul
/// operand with `ld` halves per row: 16-byte chunks XOR-swizzled within
/// each 128-byte line of banks (see `kswz` in the prelude). Rows of fewer
/// than 8 chunks share a line, so the row index is shifted to select it.
fn swz(r: &str, c: &str, ld: usize) -> String {
    let chunks = ld / 8;
    let (sh, m) = if chunks >= 8 {
        (0, 7)
    } else {
        (log2(8 / chunks.max(1)), chunks.max(1) - 1)
    };
    format!("kswz({r}, {c}, {ld}, {sh}, {m})")
}

fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

impl<'a> G<'a> {
    fn lid(&mut self, l: Layout, prio: u8) -> usize {
        if let Some(&i) = self.index.get(&l) {
            self.prio[i] = self.prio[i].max(prio);
            return i;
        }
        self.layouts.push(l.clone());
        self.prio.push(prio);
        self.index.insert(l, self.layouts.len() - 1);
        self.layouts.len() - 1
    }
    fn ty(&self, v: V) -> &'a Ty {
        &self.k.types[v]
    }
    fn vec_for(d: DType) -> usize {
        16 / d.bytes().max(1)
    }
    fn blocked(&mut self, shape: &[usize], d: DType) -> usize {
        let l = Layout::blocked(shape, self.o.warps, Self::vec_for(d));
        self.lid(l, 1)
    }

    /// The accumulator layout of a matmul producing `shape` from `dt`.
    fn acc_layout(&mut self, shape: &[usize], dt: DType) -> usize {
        let (bm, bn) = (shape[0], shape[1]);
        let w = self.o.warps;
        // Tensor-core fragments (mma.sync) are NVIDIA's: Metal takes the
        // SIMT path.
        let dt = if self.o.arch == METAL { DType::F32 } else { dt };
        if dt == DType::F16 && self.rowwise && bm / w >= 16 && bm.is_multiple_of(w) {
            let l = Layout::mma(shape, w, 1);
            let id = self.lid(l, 3);
            self.mma.insert(id);
            return id;
        }
        if dt == DType::F16 {
            // Warp tiles as square as possible: fewer fragment loads per mma.
            let mut best = None;
            let mut wm = 1;
            while wm <= w {
                let wn = w / wm;
                let (tm, tn) = (bm / wm.min(bm), bn / wn.min(bn));
                if bm.is_multiple_of(wm) && bn.is_multiple_of(wn) && tm >= 16 && tn >= 8 {
                    let score = (log2(tm) as i32 - log2(tn) as i32).abs();
                    if best.is_none_or(|(s, _, _)| score < s) {
                        best = Some((score, wm, wn));
                    }
                }
                wm *= 2;
            }
            if let Some((_, wm, wn)) = best {
                let l = Layout::mma(shape, wm, wn);
                let id = self.lid(l, 3);
                self.mma.insert(id);
                return id;
            }
        }
        let per = (bm * bn / (32 * w)).max(1);
        let tn = (per as f64).sqrt().ceil() as usize;
        let tn = tn.next_power_of_two().min(bn).min(8);
        let tm = (per / tn).max(1).min(bm);
        let l = Layout::simt(shape, w, tm, tn);
        self.lid(l, 3)
    }

    // ---------------- layout assignment ----------------

    fn assign(&mut self, b: &'a [Inst]) {
        for i in b {
            self.assign_inst(i);
        }
    }

    fn pick(&self, ops: &[V]) -> Lay {
        let mut best: Option<usize> = None;
        let mut pending = false;
        for &v in ops {
            match self.lay[v] {
                Lay::L(id) => {
                    if best.is_none_or(|b| self.prio[id] > self.prio[b]) {
                        best = Some(id);
                    }
                }
                Lay::Pending => pending = true,
                _ => {}
            }
        }
        match best {
            Some(id) => Lay::L(id),
            None if pending => Lay::Pending,
            None => Lay::Lazy,
        }
    }

    fn assign_inst(&mut self, i: &'a Inst) {
        for &o in &i.outs {
            self.def[o] = Some(i);
        }
        let out = i.outs.first().copied();
        let scalar_out = out.is_some_and(|o| self.ty(o).is_scalar());
        let l = match &i.op {
            Op::Const(_) | Op::ProgramId(_) | Op::ScalarArg(_) => Lay::Scalar,
            Op::Arange | Op::Splat(_) | Op::Broadcast(_) => Lay::Lazy,
            Op::ExpandDims(x, _) | Op::Trans(x) => match self.lay[*x] {
                Lay::L(id) => {
                    let shape = self.ty(out.unwrap()).shape.clone();
                    let nl = if matches!(i.op, Op::Trans(_)) {
                        self.layouts[id].trans()
                    } else {
                        self.layouts[id].reshape(&shape)
                    };
                    let p = self.prio[id];
                    Lay::L(self.lid(nl, p))
                }
                l => l,
            },
            Op::Unary(_, x) | Op::Cast(x) => {
                if scalar_out {
                    Lay::Scalar
                } else {
                    self.pick(&[*x])
                }
            }
            Op::Binary(_, a, b) => {
                if scalar_out {
                    Lay::Scalar
                } else {
                    self.pick(&[*a, *b])
                }
            }
            Op::Where(c, a, b) => {
                if scalar_out {
                    Lay::Scalar
                } else {
                    self.pick(&[*a, *b, *c])
                }
            }
            Op::Reduce(_, x, axis) => {
                if scalar_out {
                    Lay::Scalar
                } else {
                    let base = match self.lay[*x] {
                        Lay::L(id) => id,
                        _ => {
                            let t = self.ty(*x);
                            self.blocked(&t.shape.clone(), t.dtype)
                        }
                    };
                    let s = self.layouts[base].slice(*axis);
                    let p = self.prio[base];
                    Lay::L(self.lid(s, p))
                }
            }
            Op::Dot(a, _, _) => {
                let shape = self.ty(out.unwrap()).shape.clone();
                let dt = self.ty(*a).dtype;
                Lay::L(self.acc_layout(&shape, dt))
            }
            Op::Load { .. } => {
                let t = self.ty(out.unwrap());
                if t.is_scalar() {
                    Lay::Scalar
                } else {
                    let l = Layout::blocked(&t.shape, self.o.warps, Self::vec_for(t.dtype));
                    Lay::L(self.lid(l, 2))
                }
            }
            Op::Store { .. } => Lay::Unset,
            Op::For {
                iv,
                args,
                init,
                body,
                yields,
                ..
            } => {
                self.lay[*iv] = Lay::Scalar;
                for (a, x) in args.iter().zip(init) {
                    self.lay[*a] = match self.lay[*x] {
                        Lay::Scalar => Lay::Scalar,
                        Lay::L(id) => Lay::L(id),
                        _ => Lay::Pending,
                    };
                }
                for _ in 0..6 {
                    self.assign(body);
                    let mut changed = false;
                    for (a, y) in args.iter().zip(yields) {
                        if let Lay::L(yl) = self.lay[*y] {
                            let better = match self.lay[*a] {
                                Lay::Pending => true,
                                Lay::L(al) => al != yl && self.prio[yl] > self.prio[al],
                                _ => false,
                            };
                            if better {
                                self.lay[*a] = Lay::L(yl);
                                changed = true;
                            }
                        }
                    }
                    if !changed {
                        break;
                    }
                }
                for a in args {
                    if matches!(self.lay[*a], Lay::Pending | Lay::Lazy) {
                        let t = self.ty(*a);
                        let id = self.blocked(&t.shape.clone(), t.dtype);
                        self.lay[*a] = Lay::L(id);
                    }
                }
                self.assign(body);
                for (o, a) in i.outs.iter().zip(args) {
                    self.lay[*o] = self.lay[*a];
                }
                return;
            }
        };
        if let Some(o) = out {
            self.lay[o] = l;
        }
    }

    /// `size` positions from integer scalar `v` provably lie in `0..dim`.
    fn in_bounds(&self, v: V, size: usize, dim: usize) -> bool {
        matches!(self.rng[v], Some((lo, hi)) if lo >= 0 && hi + size as i64 <= dim as i64)
    }

    // ---------------- divisibility of integer scalars ----------------

    fn divs(&mut self, b: &[Inst]) {
        for i in b {
            let out = i.outs.first().copied();
            match &i.op {
                Op::Const(x) => {
                    let x = (*x as i64).unsigned_abs();
                    self.div[out.unwrap()] = if x == 0 { BIG } else { x.min(BIG) };
                }
                Op::Binary(op, a, b) if self.ty(out.unwrap()).is_scalar() => {
                    let (x, y) = (self.div[*a], self.div[*b]);
                    self.div[out.unwrap()] = match op {
                        Bin::Add | Bin::Sub | Bin::Max | Bin::Min => gcd(x, y),
                        Bin::Mul => (x * y).min(BIG),
                        _ => 1,
                    };
                }
                Op::For {
                    iv,
                    start,
                    step,
                    body,
                    args,
                    ..
                } => {
                    self.div[*iv] = gcd(self.div[*start], self.div[*step]);
                    for a in args {
                        self.div[*a] = 1;
                    }
                    self.divs(body);
                }
                _ => {
                    if let Some(o) = out {
                        self.div[o] = 1;
                    }
                }
            }
        }
    }

    // ---------------- emission helpers ----------------

    fn line(&mut self, s: &str) {
        for _ in 0..self.ind {
            self.out.push_str("  ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn fresh(&mut self, p: &str) -> String {
        self.n += 1;
        format!("{p}{}", self.n)
    }
    fn cached(&self, v: V, id: usize) -> Option<String> {
        self.scopes.iter().rev().find_map(|s| s.get(&(v, id)).cloned())
    }
    fn cache(&mut self, v: V, id: usize, name: String) {
        self.scopes.last_mut().unwrap().insert((v, id), name);
    }
    fn sc(&self, v: V) -> String {
        self.scal
            .get(&v)
            .cloned()
            .unwrap_or_else(|| panic!("scalar %{v} not emitted"))
    }
    /// Position of register `k` of layout `id` in the calling thread.
    fn pos(&mut self, id: usize, k: usize) -> String {
        self.used_tb.insert(id);
        let rp = self.layouts[id].reg_pos(k);
        if rp == 0 {
            format!("tb{id}")
        } else {
            format!("(tb{id} ^ {rp})")
        }
    }
    /// (row, col) of register `k` of layout `id` (row "0" for 1-D tiles).
    fn rc(&mut self, id: usize, k: usize) -> (String, String) {
        let p = self.pos(id, k);
        let l = &self.layouts[id];
        let c = l.cols();
        if l.rank() < 2 {
            ("0".into(), p)
        } else if c == 1 {
            (p, "0".into())
        } else {
            (format!("({p} >> {})", log2(c)), format!("({p} & {})", c - 1))
        }
    }
    fn decl(&mut self, dt: DType, n: usize) -> String {
        let name = self.fresh("v");
        self.line(&format!("{} {name}[{n}];", cty(dt)));
        name
    }
    /// Lane and warp bits whose basis is zero: only the threads with those
    /// bits clear need to write (the others hold the same elements).
    fn writer_pred(&self, id: usize) -> String {
        let l = &self.layouts[id];
        let mut lm = 0;
        for i in 0..5 {
            if l.lane[i] == 0 {
                lm |= 1 << i;
            }
        }
        let mut wm = 0;
        for (i, b) in l.warp.iter().enumerate() {
            if *b == 0 {
                wm |= 1 << i;
            }
        }
        let mut c = Vec::new();
        if lm != 0 {
            c.push(format!("(lane & {lm}) == 0"));
        }
        if wm != 0 {
            c.push(format!("(warp & {wm}) == 0"));
        }
        if c.is_empty() { "true".into() } else { c.join(" && ") }
    }

    fn need_scratch(&mut self, bytes: usize) {
        self.scratch = self.scratch.max(bytes);
    }

    // ---------------- materialization ----------------

    /// The C array holding `v` in layout `id`, computing or converting it
    /// here if needed.
    fn get(&mut self, v: V, id: usize) -> String {
        if let Some(n) = self.cached(v, id) {
            return n;
        }
        let name = match self.lay[v] {
            Lay::L(own) => {
                let src = self
                    .cached(v, own)
                    .unwrap_or_else(|| panic!("%{v} used before definition"));
                self.convert(&src, own, id, self.ty(v).dtype)
            }
            Lay::Lazy => self.lazy(v, id),
            l => panic!("%{v} has layout {l:?}"),
        };
        self.cache(v, id, name.clone());
        name
    }

    fn lazy(&mut self, v: V, id: usize) -> String {
        let i = self.def[v].unwrap();
        let t = self.ty(v);
        let nr = self.layouts[id].nregs();
        match &i.op {
            Op::Splat(s) => {
                let s = self.sc(*s);
                let name = self.decl(t.dtype, nr);
                for k in 0..nr {
                    self.line(&format!("{name}[{k}] = {s};"));
                }
                name
            }
            Op::Arange => {
                let name = self.decl(t.dtype, nr);
                for k in 0..nr {
                    let p = self.pos(id, k);
                    self.line(&format!("{name}[{k}] = {p};"));
                }
                name
            }
            Op::ExpandDims(x, _) => {
                let shape = self.ty(*x).shape.clone();
                let l = self.layouts[id].reshape(&shape);
                let lx = self.lid(l, 0);
                self.get(*x, lx)
            }
            Op::Trans(x) => {
                let l = self.layouts[id].trans();
                let lx = self.lid(l, 0);
                self.get(*x, lx)
            }
            Op::Broadcast(x) => {
                let shape = self.ty(*x).shape.clone();
                let p = self.layouts[id].project(&shape);
                let pid = self.lid(p.clone(), 0);
                let src = self.get(*x, pid);
                let at: HashMap<u32, usize> = (0..p.nregs()).map(|k| (p.reg_pos(k), k)).collect();
                let name = self.decl(t.dtype, nr);
                for k in 0..nr {
                    let rp = self.layouts[id].reg_pos(k);
                    let sk = at[&self.layouts[id].project_pos(&shape, rp)];
                    self.line(&format!("{name}[{k}] = {src}[{sk}];"));
                }
                name
            }
            Op::Unary(..) | Op::Binary(..) | Op::Where(..) | Op::Cast(..) => self.elementwise(i, v, id),
            _ => panic!("%{v} is not lazy"),
        }
    }

    fn convert(&mut self, src: &str, from: usize, to: usize, dt: DType) -> String {
        if from == to {
            return src.to_string();
        }
        let (a, b) = (self.layouts[from].clone(), self.layouts[to].clone());
        if let Some(map) = a.reg_map_from(&b) {
            if map.len() == a.nregs() && map.iter().enumerate().all(|(i, &m)| i == m) {
                return src.to_string();
            }
            let name = self.decl(dt, b.nregs());
            for (k, m) in map.iter().enumerate() {
                self.line(&format!("{name}[{k}] = {src}[{m}];"));
            }
            return name;
        }
        // Through shared memory.
        self.need_scratch(a.numel() * 4);
        let buf = if dt == DType::F32 || dt == DType::F16 {
            "tsf"
        } else {
            "tsi"
        };
        self.line("KSYNC();");
        let pred = self.writer_pred(from);
        self.line(&format!("if ({pred}) {{"));
        self.ind += 1;
        for k in 0..a.nregs() {
            let p = self.pos(from, k);
            self.line(&format!("{buf}[{p}] = {src}[{k}];"));
        }
        self.ind -= 1;
        self.line("}");
        self.line("KSYNC();");
        let name = self.decl(dt, b.nregs());
        for k in 0..b.nregs() {
            let p = self.pos(to, k);
            self.line(&format!("{name}[{k}] = {buf}[{p}];"));
        }
        name
    }

    // ---------------- operations ----------------

    fn un(op: Un, x: &str, d: DType) -> String {
        let f = d.is_float();
        match op {
            Un::Neg => format!("(-{x})"),
            Un::Exp => format!("kexp({x})"),
            Un::Log => format!("logf({x})"),
            Un::Sqrt => format!("sqrtf({x})"),
            Un::Rsqrt => format!("rsqrtf({x})"),
            Un::Abs => {
                if f {
                    format!("fabsf({x})")
                } else {
                    format!("({x} < 0 ? -{x} : {x})")
                }
            }
            Un::Not => format!("(!{x})"),
        }
    }

    fn bin(op: Bin, a: &str, b: &str, d: DType) -> String {
        let f = d.is_float();
        match op {
            Bin::Add => format!("({a} + {b})"),
            Bin::Sub => format!("({a} - {b})"),
            Bin::Mul => format!("({a} * {b})"),
            Bin::Div => format!("({a} / {b})"),
            Bin::FloorDiv => format!("kfdiv({a}, {b})"),
            Bin::Mod => format!("kfmod({a}, {b})"),
            Bin::Max => {
                if f {
                    format!("fmaxf({a}, {b})")
                } else {
                    format!("kmaxi({a}, {b})")
                }
            }
            Bin::Min => {
                if f {
                    format!("fminf({a}, {b})")
                } else {
                    format!("kmini({a}, {b})")
                }
            }
            Bin::Lt => format!("({a} < {b})"),
            Bin::Le => format!("({a} <= {b})"),
            Bin::Gt => format!("({a} > {b})"),
            Bin::Ge => format!("({a} >= {b})"),
            Bin::Eq => format!("({a} == {b})"),
            Bin::Ne => format!("({a} != {b})"),
            Bin::And => format!("({a} && {b})"),
            Bin::Or => format!("({a} || {b})"),
        }
    }

    /// `e` (computed in f32 or int) as a value of dtype `d`.
    fn fit(e: String, from: DType, d: DType) -> String {
        match (from, d) {
            (_, DType::F16) => format!("kr16({e})"),
            (DType::F32 | DType::F16, DType::I32) => format!("(int)({e})"),
            (_, DType::Bool) if from != DType::Bool => format!("({e} != 0)"),
            (DType::I32 | DType::Bool, DType::F32) => format!("(float)({e})"),
            _ => e,
        }
    }

    /// The expression for operation `i` given its operand expressions.
    fn expr_of(&self, i: &Inst, args: &[String]) -> String {
        let out = i.outs[0];
        let d = self.ty(out).dtype;
        match &i.op {
            Op::Unary(op, x) => Self::fit(Self::un(*op, &args[0], self.ty(*x).dtype), self.ty(*x).dtype, d),
            Op::Binary(op, a, _) => {
                let ad = self.ty(*a).dtype;
                let e = Self::bin(*op, &args[0], &args[1], ad);
                if op.is_compare() || matches!(op, Bin::And | Bin::Or) {
                    e
                } else {
                    Self::fit(e, ad, d)
                }
            }
            Op::Where(..) => format!("({} ? {} : {})", args[0], args[1], args[2]),
            Op::Cast(x) => Self::fit(args[0].clone(), self.ty(*x).dtype, d),
            _ => unreachable!(),
        }
    }

    fn operands(op: &Op) -> Vec<V> {
        match op {
            Op::Unary(_, x) | Op::Cast(x) => vec![*x],
            Op::Binary(_, a, b) => vec![*a, *b],
            Op::Where(c, a, b) => vec![*c, *a, *b],
            _ => vec![],
        }
    }

    fn elementwise(&mut self, i: &Inst, v: V, id: usize) -> String {
        let ops = Self::operands(&i.op);
        let names: Vec<String> = ops.iter().map(|&o| self.get(o, id)).collect();
        let nr = self.layouts[id].nregs();
        let name = self.decl(self.ty(v).dtype, nr);
        for k in 0..nr {
            let a: Vec<String> = names.iter().map(|n| format!("{n}[{k}]")).collect();
            let e = self.expr_of(i, &a);
            self.line(&format!("{name}[{k}] = {e};"));
        }
        name
    }

    fn gen_block(&mut self, b: &'a [Inst]) {
        for i in b {
            self.gen_inst(i);
        }
    }

    fn gen_inst(&mut self, i: &'a Inst) {
        let out = i.outs.first().copied();
        match &i.op {
            Op::Const(x) => {
                let o = out.unwrap();
                let d = self.ty(o).dtype;
                let n = self.fresh("s");
                self.line(&format!("const {} {n} = {};", cty(d), lit(*x, d)));
                self.scal.insert(o, n);
            }
            Op::ProgramId(a) => {
                let n = self.fresh("s");
                self.line(&format!("const int {n} = (int)blockIdx.{};", ["x", "y", "z"][*a]));
                self.scal.insert(out.unwrap(), n);
            }
            Op::ScalarArg(p) => {
                self.scal.insert(out.unwrap(), format!("a{p}"));
            }
            Op::Unary(..) | Op::Binary(..) | Op::Where(..) | Op::Cast(..) => {
                let o = out.unwrap();
                match self.lay[o] {
                    Lay::Scalar => {
                        let args: Vec<String> = Self::operands(&i.op).iter().map(|&x| self.sc(x)).collect();
                        let e = self.expr_of(i, &args);
                        let n = self.fresh("s");
                        self.line(&format!("const {} {n} = {e};", cty(self.ty(o).dtype)));
                        self.scal.insert(o, n);
                    }
                    Lay::L(id) => {
                        let n = self.elementwise(i, o, id);
                        self.cache(o, id, n);
                    }
                    _ => {}
                }
            }
            Op::ExpandDims(x, _) | Op::Trans(x) => {
                let o = out.unwrap();
                // A transposed packed load is staged by the matmul itself.
                if self.raw.contains_key(x) {
                    return;
                }
                if let (Lay::L(id), Lay::L(xid)) = (self.lay[o], self.lay[*x]) {
                    let n = self.get(*x, xid);
                    self.cache(o, id, n);
                }
            }
            Op::Arange | Op::Splat(_) | Op::Broadcast(_) => {}
            Op::Reduce(op, x, axis) => self.gen_reduce(out.unwrap(), *op, *x, *axis),
            Op::Dot(a, b, c) => self.gen_dot(out.unwrap(), *a, *b, *c),
            Op::Load { arr, idx, other } => {
                if !self.prefetched.contains(&out.unwrap()) {
                    self.gen_load(out.unwrap(), *arr, idx, *other)
                }
            }
            Op::Store { arr, idx, value } => self.gen_store(*arr, idx, *value),
            Op::For {
                iv,
                start,
                end,
                step,
                args,
                init,
                body,
                yields,
            } => {
                let mut names = Vec::new();
                for (a, x) in args.iter().zip(init) {
                    let t = self.ty(*a);
                    match self.lay[*a] {
                        Lay::Scalar => {
                            let n = self.fresh("s");
                            let e = self.sc(*x);
                            self.line(&format!("{} {n} = {e};", cty(t.dtype)));
                            self.scal.insert(*a, n.clone());
                            names.push(n);
                        }
                        Lay::L(id) => {
                            let src = self.get(*x, id);
                            let nr = self.layouts[id].nregs();
                            let n = self.decl(t.dtype, nr);
                            for k in 0..nr {
                                self.line(&format!("{n}[{k}] = {src}[{k}];"));
                            }
                            names.push(n);
                        }
                        l => panic!("loop argument with layout {l:?}"),
                    }
                }
                let ivn = self.fresh("s");
                let (s, e, st) = (self.sc(*start), self.sc(*end), self.sc(*step));
                // Software pipeline: loads that depend only on the loop
                // index are issued one iteration ahead, so they are in
                // flight while this iteration computes.
                let pf = self.prefetchable(body, *iv, args);
                let mut pipes = Vec::new();
                for li in &pf {
                    let Op::Load { arr, idx, other } = &li.op else {
                        unreachable!()
                    };
                    let o = li.outs[0];
                    let raw = self.raw.contains_key(&o);
                    let name = self.decl_load(o);
                    let ix = self.ix_subst(idx, *iv, &format!("({s})"), body, args);
                    // Guarded by the loop condition: index ranges hold only
                    // for indices the loop runs.
                    let guard = format!("({s}) < ({e})");
                    self.load_into(&name, raw, o, *arr, &ix, *other, Some(&guard));
                    pipes.push((*li, name, raw));
                }
                self.prestage(body);
                let par = self.fresh("p");
                self.line(&format!("int {par} = 0;"));
                self.scal.insert(*iv, ivn.clone());
                self.line(&format!("for (int {ivn} = {s}; {ivn} < {e}; {ivn} += {st}) {{"));
                self.loops.push(par.clone());
                self.ind += 1;
                self.scopes.push(HashMap::new());
                for (a, n) in args.iter().zip(&names) {
                    if let Lay::L(id) = self.lay[*a] {
                        self.cache(*a, id, n.clone());
                    }
                }
                for (li, name, raw) in &pipes {
                    let o = li.outs[0];
                    let cur = self.decl_load(o);
                    let nr = self.layouts[self.lid_of(o)].nregs();
                    let n = if *raw { nr / 2 } else { nr };
                    for k in 0..n {
                        self.line(&format!("{cur}[{k}] = {name}[{k}];"));
                    }
                    let key = if *raw { RAW } else { self.lid_of(o) };
                    self.cache(o, key, cur);
                    self.prefetched.insert(o);
                }
                for (li, name, raw) in &pipes {
                    let Op::Load { arr, idx, other } = &li.op else {
                        unreachable!()
                    };
                    let ix = self.ix_subst(idx, *iv, &format!("({ivn} + {st})"), body, args);
                    let guard = format!("({ivn} + {st}) < ({e})");
                    self.load_into(name, *raw, li.outs[0], *arr, &ix, *other, Some(&guard));
                }
                self.gen_block(body);
                // Yields: every new value first, then the assignments.
                let mut tmps = Vec::new();
                for (a, y) in args.iter().zip(yields) {
                    match self.lay[*a] {
                        Lay::Scalar => {
                            let t = self.fresh("s");
                            let e = self.sc(*y);
                            self.line(&format!("const {} {t} = {e};", cty(self.ty(*a).dtype)));
                            tmps.push(t);
                        }
                        Lay::L(id) => tmps.push(self.get(*y, id)),
                        _ => unreachable!(),
                    }
                }
                for ((a, n), t) in args.iter().zip(&names).zip(&tmps) {
                    if n == t {
                        continue;
                    }
                    match self.lay[*a] {
                        Lay::Scalar => self.line(&format!("{n} = {t};")),
                        Lay::L(id) => {
                            for k in 0..self.layouts[id].nregs() {
                                self.line(&format!("{n}[{k}] = {t}[{k}];"));
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                self.line(&format!("{par} ^= 1;"));
                self.loops.pop();
                self.scopes.pop();
                self.ind -= 1;
                self.line("}");
                for ((o, a), n) in i.outs.iter().zip(args).zip(&names) {
                    match self.lay[*a] {
                        Lay::Scalar => {
                            self.scal.insert(*o, n.clone());
                        }
                        Lay::L(id) => self.cache(*o, id, n.clone()),
                        _ => {}
                    }
                }
            }
        }
    }

    fn gen_reduce(&mut self, out: V, op: Red, x: V, axis: usize) {
        let xt = self.ty(x);
        let lx = match self.lay[x] {
            Lay::L(id) => id,
            _ => self.blocked(&xt.shape.clone(), xt.dtype),
        };
        let src = self.get(x, lx);
        let l = self.layouts[lx].clone();
        let s = l.slice(axis);
        let d = self.ty(out).dtype;
        let comb: fn(&str, &str, DType) -> String = match op {
            Red::Sum => |a, b, _| format!("({a} + {b})"),
            Red::Max => |a, b, d| {
                if d.is_float() {
                    format!("fmaxf({a}, {b})")
                } else {
                    format!("kmaxi({a}, {b})")
                }
            },
            Red::Min => |a, b, d| {
                if d.is_float() {
                    format!("fminf({a}, {b})")
                } else {
                    format!("kmini({a}, {b})")
                }
            },
        };
        // Registers: combine those that land on the same output register.
        let at: HashMap<u32, usize> = (0..s.nregs()).map(|k| (s.reg_pos(k), k)).collect();
        let mut groups = vec![Vec::new(); s.nregs()];
        for k in 0..l.nregs() {
            groups[at[&l.slice_pos(axis, l.reg_pos(k))]].push(k);
        }
        let name = self.decl(d, s.nregs());
        for (j, g) in groups.iter().enumerate() {
            let mut e = format!("{src}[{}]", g[0]);
            for k in &g[1..] {
                e = comb(&e, &format!("{src}[{k}]"), d);
            }
            self.line(&format!("{name}[{j}] = {e};"));
        }
        // Lanes: shuffle across the lanes that hold other positions along the axis.
        let (lanes, warps) = l.axis_bits(axis);
        let sh = if d.is_float() { "kshfl_xor" } else { "kshfl_xor_i" };
        for b in lanes {
            for j in 0..s.nregs() {
                let e = comb(&format!("{name}[{j}]"), &format!("{sh}({name}[{j}], {})", 1 << b), d);
                self.line(&format!("{name}[{j}] = {e};"));
            }
        }
        // Warps: through shared memory.
        if !warps.is_empty() {
            let nw = 1usize << warps.len();
            let sn = s.numel();
            self.need_scratch(nw * sn * 4);
            let buf = if d.is_float() { "tsf" } else { "tsi" };
            let wa: Vec<String> = warps
                .iter()
                .enumerate()
                .map(|(i, b)| format!("(((warp >> {b}) & 1) << {i})"))
                .collect();
            let wa = wa.join(" | ");
            let sid = self.lid(s.clone(), 0);
            self.line("KSYNC();");
            for j in 0..s.nregs() {
                let p = self.pos(sid, j);
                self.line(&format!("{buf}[({wa}) * {sn} + {p}] = {name}[{j}];"));
            }
            self.line("KSYNC();");
            for j in 0..s.nregs() {
                let p = self.pos(sid, j);
                let mut e = format!("{buf}[{p}]");
                for w in 1..nw {
                    e = comb(&e, &format!("{buf}[{} + {p}]", w * sn), d);
                }
                self.line(&format!("{name}[{j}] = {e};"));
            }
        }
        match self.lay[out] {
            Lay::Scalar => {
                let n = self.fresh("s");
                self.line(&format!("const {} {n} = {name}[0];", cty(d)));
                self.scal.insert(out, n);
            }
            Lay::L(id) => {
                let sid = self.lid(s, 0);
                let n = self.convert(&name, sid, id, d);
                self.cache(out, id, n);
            }
            _ => unreachable!(),
        }
    }

    /// The C expressions of a block's indices: (slice size, or None for a
    /// point; the start or index expression; its value, for divisibility).
    fn ix_exprs(&self, idx: &[Ix]) -> Vec<(Option<usize>, String, V)> {
        idx.iter()
            .map(|ix| match ix {
                Ix::Point(p) => (None, self.sc(*p), *p),
                Ix::Slice(s, n) => (Some(*n), self.sc(*s), *s),
            })
            .collect()
    }

    /// Emits a block access's scalar parts: (base offset, in-bounds
    /// condition of the point indices, and per slice dimension its (array
    /// stride, start expression, array dimension, start value)).
    fn block_access(&mut self, arr: usize, ix: &[(Option<usize>, String, V)]) -> Access {
        let PKind::Array { dims, .. } = &self.k.params[arr].kind else {
            unreachable!()
        };
        let st = strides(dims);
        let mut base = Vec::new();
        let mut cond = Vec::new();
        let mut sl = Vec::new();
        for (d, (size, e, v)) in ix.iter().enumerate() {
            base.push(format!("(KI64)({e}) * {}", st[d]));
            match size {
                None => {
                    if !self.in_bounds(*v, 1, dims[d]) {
                        cond.push(format!("(unsigned)({e}) < {}u", dims[d]))
                    }
                }
                Some(_) => sl.push((st[d], e.clone(), dims[d], *v)),
            }
        }
        let b = self.fresh("o");
        self.line(&format!("const KI64 {b} = {};", base.join(" + ")));
        let c = self.fresh("m");
        self.line(&format!(
            "const bool {c} = {};",
            if cond.is_empty() {
                "true".into()
            } else {
                cond.join(" && ")
            }
        ));
        (b, c, sl)
    }

    /// The vector width usable for registers of layout `id` reading or
    /// writing a block whose last dimension has `stride`, array dimension
    /// `dim` and start value `sv`.
    fn vec_width(&self, id: usize, dt: DType, stride: usize, dim: usize, sv: V) -> usize {
        if stride != 1 || !matches!(dt, DType::F32 | DType::F16) {
            return 1;
        }
        let l = &self.layouts[id];
        let mut v = 1;
        while v < l.cols() && (log2(v) as usize) < l.reg.len() && l.reg[log2(v) as usize] == v as u32 {
            v *= 2;
        }
        let mut v = v.min(16 / dt.bytes());
        while v > 1 && !(dim.is_multiple_of(v) && self.div[sv].is_multiple_of(v as u64)) {
            v /= 2;
        }
        v
    }

    /// The vector width of a tile load (from static information only).
    fn load_vec(&self, out: V, arr: usize, idx: &[Ix]) -> usize {
        let PKind::Array { dims, dtype } = &self.k.params[arr].kind else {
            unreachable!()
        };
        let Lay::L(id) = self.lay[out] else { return 1 };
        let st = strides(dims);
        match idx.iter().enumerate().rev().find(|(_, ix)| matches!(ix, Ix::Slice(..))) {
            Some((d, Ix::Slice(s, _))) => self.vec_width(id, *dtype, st[d], dims[d], *s),
            _ => 1,
        }
    }

    fn elem(&mut self, id: usize, k: usize, sl: &[SliceDim]) -> (String, String) {
        let (r, c) = self.rc(id, k);
        let coords: Vec<String> = if sl.len() == 2 { vec![r, c] } else { vec![c] };
        let sizes: Vec<usize> = if sl.len() == 2 {
            self.layouts[id].shape.clone()
        } else {
            vec![self.layouts[id].cols()]
        };
        let mut off = Vec::new();
        let mut cond = Vec::new();
        for (((st, s, dim, sv), x), size) in sl.iter().zip(&coords).zip(&sizes) {
            off.push(if *st == 1 { x.clone() } else { format!("{x} * {st}") });
            if !self.in_bounds(*sv, *size, *dim) {
                cond.push(format!("(unsigned)({s} + {x}) < {dim}u"));
            }
        }
        (off.join(" + "), cond.join(" && "))
    }

    fn arr_ptr(&self, arr: usize) -> String {
        format!("a{arr}")
    }

    fn lid_of(&self, v: V) -> usize {
        match self.lay[v] {
            Lay::L(id) => id,
            l => panic!("%{v} has no layout ({l:?})"),
        }
    }

    /// Declares the array a tile load fills: packed halves or values.
    fn decl_load(&mut self, out: V) -> String {
        let id = self.lid_of(out);
        let nr = self.layouts[id].nregs();
        if self.raw.contains_key(&out) {
            let name = self.fresh("r");
            self.line(&format!("unsigned {name}[{}];", nr / 2));
            name
        } else {
            self.decl(self.ty(out).dtype, nr)
        }
    }

    fn gen_load(&mut self, out: V, arr: usize, idx: &[Ix], other: f64) {
        let PKind::Array { dtype, .. } = self.k.params[arr].kind else {
            unreachable!()
        };
        let ix = self.ix_exprs(idx);
        if self.ty(out).is_scalar() {
            let p = self.arr_ptr(arr);
            let (b, m, _) = self.block_access(arr, &ix);
            let n = self.fresh("s");
            let o = lit(other, dtype);
            let rd = match dtype {
                DType::F16 => format!("kh2f({p}[{b}])"),
                DType::Bool => format!("({p}[{b}] != 0)"),
                _ => format!("{p}[{b}]"),
            };
            self.line(&format!("const {} {n} = {m} ? {rd} : {o};", cty(dtype)));
            self.scal.insert(out, n);
            return;
        }
        let raw = self.raw.contains_key(&out);
        let name = self.decl_load(out);
        self.load_into(&name, raw, out, arr, &ix, other, None);
        let key = if raw { RAW } else { self.lid_of(out) };
        self.cache(out, key, name);
    }

    /// Fills `name` with the tile load `out` at indices `ix`.
    #[allow(clippy::too_many_arguments)]
    fn load_into(
        &mut self,
        name: &str,
        raw: bool,
        out: V,
        arr: usize,
        ix: &[(Option<usize>, String, V)],
        other: f64,
        guard: Option<&str>,
    ) {
        let PKind::Array { dtype, .. } = self.k.params[arr].kind else {
            unreachable!()
        };
        let id = self.lid_of(out);
        let p = self.arr_ptr(arr);
        let (b, m, sl) = self.block_access(arr, ix);
        let m = match guard {
            Some(g) => format!("({m} && ({g}))"),
            None => m,
        };
        let nr = self.layouts[id].nregs();
        let o = lit(other, dtype);
        let v = match sl.last() {
            Some((st, _, dim, sv)) => self.vec_width(id, dtype, *st, *dim, *sv),
            None => 1,
        };
        assert!(!raw || v >= 2, "packed load without vectors");
        let rd = |e: String| match dtype {
            DType::F16 => format!("kh2f({e})"),
            DType::Bool => format!("({e} != 0)"),
            _ => e,
        };
        let mut k = 0;
        while k < nr {
            let (off, cond) = self.elem(id, k, &sl);
            let cond = if cond.is_empty() {
                m.clone()
            } else {
                format!("{m} && {cond}")
            };
            let addr = format!("{p} + {b} + {off}");
            if v == 1 {
                self.line(&format!("{name}[{k}] = ({cond}) ? {} : {o};", rd(format!("*({addr})"))));
                k += 1;
                continue;
            }
            let tn = self.fresh("t");
            self.line(&format!("if ({cond}) {{"));
            self.ind += 1;
            let (parts, packed): (Vec<String>, Vec<String>) = match (dtype, v) {
                (DType::F32, 2) => {
                    self.line(&format!("const float2 {tn} = kld2f({addr});"));
                    (vec![format!("{tn}.x"), format!("{tn}.y")], vec![])
                }
                (DType::F32, 4) => {
                    self.line(&format!("const kf4 {tn} = kld4({addr});"));
                    (
                        ["x", "y", "z", "w"].iter().map(|c| format!("{tn}.{c}")).collect(),
                        vec![],
                    )
                }
                (DType::F16, 2) => {
                    self.line(&format!("const unsigned {tn} = kld2({addr});"));
                    (vec![format!("klo({tn})"), format!("khi({tn})")], vec![tn.clone()])
                }
                (DType::F16, 4) => {
                    self.line(&format!("const uint2 {tn} = kld4h({addr});"));
                    let c = ["x", "y"];
                    (
                        c.iter()
                            .flat_map(|c| [format!("klo({tn}.{c})"), format!("khi({tn}.{c})")])
                            .collect(),
                        c.iter().map(|c| format!("{tn}.{c}")).collect(),
                    )
                }
                (DType::F16, 8) => {
                    self.line(&format!("const kh8 {tn} = kld8h({addr});"));
                    let c = ["x", "y", "z", "w"];
                    (
                        c.iter()
                            .flat_map(|c| [format!("klo({tn}.{c})"), format!("khi({tn}.{c})")])
                            .collect(),
                        c.iter().map(|c| format!("{tn}.{c}")).collect(),
                    )
                }
                _ => unreachable!("vector {v} of {dtype:?}"),
            };
            if raw {
                for (j, e) in packed.iter().enumerate() {
                    self.line(&format!("{name}[{}] = {e};", k / 2 + j));
                }
            } else {
                for (j, e) in parts.iter().enumerate() {
                    self.line(&format!("{name}[{}] = {e};", k + j));
                }
            }
            self.ind -= 1;
            self.line("} else {");
            self.ind += 1;
            if raw {
                for j in 0..v / 2 {
                    self.line(&format!("{name}[{}] = kpack({o}, {o});", k / 2 + j));
                }
            } else {
                for j in 0..v {
                    self.line(&format!("{name}[{}] = {o};", k + j));
                }
            }
            self.ind -= 1;
            self.line("}");
            k += v;
        }
    }

    fn gen_store(&mut self, arr: usize, idx: &[Ix], value: V) {
        let PKind::Array { dtype, .. } = self.k.params[arr].kind else {
            unreachable!()
        };
        let t = self.ty(value);
        let p = self.arr_ptr(arr);
        let ix = self.ix_exprs(idx);
        let wr = |e: &str| match dtype {
            DType::F16 => format!("kf2h({e})"),
            DType::Bool => format!("(unsigned char)({e})"),
            _ => e.to_string(),
        };
        if t.is_scalar() {
            let (b, m, _) = self.block_access(arr, &ix);
            let s = self.sc(value);
            self.line(&format!("if ({m} && threadIdx.x == 0) {p}[{b}] = {};", wr(&s)));
            return;
        }
        let id = match self.lay[value] {
            Lay::L(id) => id,
            _ => self.blocked(&t.shape.clone(), t.dtype),
        };
        let src = self.get(value, id);
        let (b, m, sl) = self.block_access(arr, &ix);
        let pred = self.writer_pred(id);
        self.line(&format!("if ({pred}) {{"));
        self.ind += 1;
        let nr = self.layouts[id].nregs();
        let v = match sl.last() {
            Some((st, _, dim, sv)) => self.vec_width(id, dtype, *st, *dim, *sv),
            None => 1,
        };
        let mut k = 0;
        while k < nr {
            let (off, cond) = self.elem(id, k, &sl);
            let cond = if cond.is_empty() {
                m.clone()
            } else {
                format!("{m} && {cond}")
            };
            let addr = format!("{p} + {b} + {off}");
            let vals: Vec<String> = (0..v).map(|j| format!("{src}[{}]", k + j)).collect();
            let st = match (dtype, v) {
                (_, 1) => format!("*({addr}) = {};", wr(&vals[0])),
                (DType::F32, 2) => format!("kst2f({addr}, {});", vals.join(", ")),
                (DType::F32, 4) => format!("kst4f({addr}, {});", vals.join(", ")),
                (DType::F16, 2) => format!("kst2h({addr}, {});", vals.join(", ")),
                (DType::F16, 4) => format!("kst4hf({addr}, {});", vals.join(", ")),
                (DType::F16, 8) => format!("kst8hf({addr}, {});", vals.join(", ")),
                _ => unreachable!(),
            };
            self.line(&format!("if ({cond}) {st}"));
            k += v;
        }
        self.ind -= 1;
        self.line("}");
    }

    // ---------------- matmuls ----------------

    /// f16 loads whose every use is as a tensor-core matmul operand (directly
    /// or transposed) stay packed: two halves per register, staged into
    /// shared memory 16 bytes at a time.
    fn find_raw(&mut self) {
        fn walk<'b>(b: &'b [Inst], f: &mut dyn FnMut(&'b Inst)) {
            for i in b {
                f(i);
                if let Op::For { body, .. } = &i.op {
                    walk(body, f);
                }
            }
        }
        let n = self.k.types.len();
        let mut uses = vec![0usize; n];
        let mut mma_uses = vec![0usize; n];
        let mut trans_of: HashMap<V, V> = HashMap::new();
        let mut loads = Vec::new();
        let body = &self.k.body;
        let lay = self.lay.clone();
        let mma = self.mma.clone();
        walk(body, &mut |i| {
            for v in all_operands(&i.op) {
                uses[v] += 1;
            }
            match &i.op {
                Op::Dot(a, b, _) => {
                    if matches!(lay[i.outs[0]], Lay::L(id) if mma.contains(&id)) {
                        mma_uses[*a] += 1;
                        mma_uses[*b] += 1;
                    }
                }
                Op::Trans(x) => {
                    trans_of.insert(i.outs[0], *x);
                }
                Op::Load { arr, idx, .. }
                    if !self.k.types[i.outs[0]].is_scalar() && self.k.types[i.outs[0]].dtype == DType::F16 =>
                {
                    loads.push((i.outs[0], *arr, idx.clone()));
                }
                _ => {}
            }
        });
        let mut via = vec![0usize; n];
        for (t, x) in &trans_of {
            if uses[*t] > 0 && uses[*t] == mma_uses[*t] {
                via[*x] += uses[*t];
            }
        }
        for (o, arr, idx) in loads {
            let v = self.load_vec(o, arr, &idx);
            if uses[o] > 0 && uses[o] == mma_uses[o] + via[o] && v >= 2 {
                self.raw.insert(o, v);
            }
        }
    }

    /// A matmul operand as staged: (array, layout, packed, transposed): a
    /// packed load (or a transposed one), or registers of any layout.
    fn operand(&mut self, x: V) -> (String, usize, bool, bool) {
        if let Some(Inst { op: Op::Trans(y), .. }) = self.def[x]
            && self.raw.contains_key(y)
        {
            let n = self.cached(*y, RAW).expect("packed operand");
            return (n, self.lid_of(*y), true, true);
        }
        if self.raw.contains_key(&x) {
            let n = self.cached(x, RAW).expect("packed operand");
            return (n, self.lid_of(x), true, false);
        }
        let t = self.ty(x);
        let id = match self.lay[x] {
            Lay::L(id) => id,
            _ => self.blocked(&t.shape.clone(), t.dtype),
        };
        (self.get(x, id), id, false, false)
    }

    /// Writes an operand tile into shared memory `s` (row-major, `ld`
    /// halves per row) in the orientation its registers hold it.
    fn stage(&mut self, s: &str, ld: usize, name: &str, id: usize, raw: Option<usize>) {
        let pred = self.writer_pred(id);
        self.line(&format!("if ({pred}) {{"));
        self.ind += 1;
        let nr = self.layouts[id].nregs();
        match raw {
            Some(v) => {
                let mut k = 0;
                while k < nr {
                    let (r, c) = self.rc(id, k);
                    let at = format!("{s} + {}", swz(&r.to_string(), &c.to_string(), ld));
                    if v == 8 {
                        let w: Vec<String> = (0..4).map(|j| format!("{name}[{}]", k / 2 + j)).collect();
                        self.line(&format!("kst16({at}, {});", w.join(", ")));
                    } else {
                        for j in 0..v / 2 {
                            self.line(&format!("kstu({at} + {}, {name}[{}]);", 2 * j, k / 2 + j));
                        }
                    }
                    k += v;
                }
            }
            None => {
                for k in 0..nr {
                    let (r, c) = self.rc(id, k);
                    let at = swz(&r.to_string(), &c.to_string(), ld);
                    self.line(&format!("{s}[{at}] = kf2h({name}[{k}]);"));
                }
            }
        }
        self.ind -= 1;
        self.line("}");
    }

    fn gen_dot(&mut self, out: V, a: V, b: V, c: V) {
        let id = self.lid_of(out);
        let (ta, tb) = (self.ty(a), self.ty(b));
        let (bm, bk, bn) = (ta.shape[0], ta.shape[1], tb.shape[1]);
        let cv = self.get(c, id);
        let nr = self.layouts[id].nregs();
        let acc = self.decl(DType::F32, nr);
        for k in 0..nr {
            self.line(&format!("{acc}[{k}] = {cv}[{k}];"));
        }
        if self.mma.contains(&id) {
            let l = self.layouts[id].clone();
            let wbits_m = l.warp.iter().filter(|&&w| w >= bn as u32).count();
            let wm = 1usize << wbits_m;
            let wn = self.o.warps / wm;
            let (tm, tn) = (bm / wm, bn / wn);
            let (nm, nn) = (tm / 16, tn / 8);
            let areg = self.reg_operand(a, id);
            let mut staged = false;
            let pa = if areg.is_none() {
                Some(self.place(a, true, &mut staged))
            } else {
                None
            };
            let (sb, ldb, btr) = self.place(b, false, &mut staged);
            if staged {
                self.line("KSYNC();");
            }
            self.line("{");
            self.ind += 1;
            self.line(&format!(
                "const int r0 = (warp & {}) * {tm}, c0 = (warp >> {wbits_m}) * {tn};",
                wm - 1
            ));
            let nna = bk / 8;
            for kk in (0..bk).step_by(16) {
                self.line("{");
                self.ind += 1;
                self.line(&format!("unsigned fa[{}], fb[{}];", 4 * nm, 2 * nn));
                for mi in 0..nm {
                    match (&areg, &pa) {
                        (Some(an), _) => {
                            // An accumulator-layout tile is already an A fragment.
                            let (j0, j1) = (4 * (kk / 8 + nna * mi), 4 * (kk / 8 + 1 + nna * mi));
                            self.line(&format!(
                                "fa[{}] = kpack({an}[{j0}], {an}[{}]); fa[{}] = kpack({an}[{}], {an}[{}]); fa[{}] = kpack({an}[{j1}], {an}[{}]); fa[{}] = kpack({an}[{}], {an}[{}]);",
                                4 * mi, j0 + 1, 4 * mi + 1, j0 + 2, j0 + 3, 4 * mi + 2, j1 + 1, 4 * mi + 3, j1 + 2, j1 + 3
                            ));
                        }
                        (None, Some((sa, lda, true))) => self.line(&format!(
                            "kldm4t(&fa[{}], {sa} + {});",
                            4 * mi,
                            swz(
                                &format!("{kk} + (lane >> 4) * 8 + (lane & 7)"),
                                &format!("r0 + {} + ((lane >> 3) & 1) * 8", mi * 16),
                                *lda
                            )
                        )),
                        (None, Some((sa, lda, false))) => self.line(&format!(
                            "kldm4(&fa[{}], {sa} + {});",
                            4 * mi,
                            swz(
                                &format!("r0 + {} + (lane & 15)", mi * 16),
                                &format!("{kk} + (lane >> 4) * 8"),
                                *lda
                            )
                        )),
                        _ => unreachable!(),
                    }
                }
                let mut ni = 0;
                while ni < nn {
                    let pair = ni + 1 < nn;
                    let n0 = ni * 8;
                    let (f, r, c) = match (btr, pair) {
                        (false, true) => (
                            "kldm4t",
                            format!("{kk} + ((lane >> 3) & 1) * 8 + (lane & 7)"),
                            format!("c0 + {n0} + (lane >> 4) * 8"),
                        ),
                        (false, false) => (
                            "kldm2t",
                            format!("{kk} + ((lane >> 3) & 1) * 8 + (lane & 7)"),
                            format!("c0 + {n0}"),
                        ),
                        (true, true) => (
                            "kldm4",
                            format!("c0 + {n0} + (lane >> 4) * 8 + (lane & 7)"),
                            format!("{kk} + ((lane >> 3) & 1) * 8"),
                        ),
                        (true, false) => (
                            "kldm2",
                            format!("c0 + {n0} + (lane & 7)"),
                            format!("{kk} + ((lane >> 3) & 1) * 8"),
                        ),
                    };
                    let addr = format!("{sb} + {}", swz(&r, &c, ldb));
                    self.line(&format!("{f}(&fb[{}], {addr});", 2 * ni));
                    ni += if pair { 2 } else { 1 };
                }
                for mi in 0..nm {
                    for ni in 0..nn {
                        let at = 4 * (ni + nn * mi);
                        self.line(&format!(
                            "kmma(&{acc}[{at}], fa[{}], fa[{}], fb[{}]);",
                            4 * mi,
                            4 * mi + 1,
                            2 * ni
                        ));
                        self.line(&format!(
                            "kmma(&{acc}[{at}], fa[{}], fa[{}], fb[{}]);",
                            4 * mi + 2,
                            4 * mi + 3,
                            2 * ni + 1
                        ));
                    }
                }
                self.ind -= 1;
                self.line("}");
            }
            self.ind -= 1;
            self.line("}");
        } else {
            // CUDA cores: A as [k][m], B as [k][n], in f32.
            let la = match self.lay[a] {
                Lay::L(x) => x,
                _ => self.blocked(&ta.shape.clone(), ta.dtype),
            };
            let lb = match self.lay[b] {
                Lay::L(x) => x,
                _ => self.blocked(&tb.shape.clone(), tb.dtype),
            };
            let av = self.get(a, la);
            let bv = self.get(b, lb);
            let (lda, ldb) = (bm + 4, bn + 4);
            let bytes_a = bk * lda * 4;
            let base = self.alloc(bytes_a + bk * ldb * 4);
            let (sa, sb) = (self.fresh("As"), self.fresh("Bs"));
            self.line("KSYNC();");
            self.line(&format!("TSP float *{sa} = (TSP float *)(tsmem + TSO_DOT + {base});"));
            self.line(&format!(
                "TSP float *{sb} = (TSP float *)(tsmem + TSO_DOT + {});",
                base + bytes_a
            ));
            for k in 0..self.layouts[la].nregs() {
                let (r, cc) = self.rc(la, k);
                self.line(&format!("{sa}[{cc} * {lda} + {r}] = {av}[{k}];"));
            }
            for k in 0..self.layouts[lb].nregs() {
                let (r, cc) = self.rc(lb, k);
                self.line(&format!("{sb}[{r} * {ldb} + {cc}] = {bv}[{k}];"));
            }
            self.line("KSYNC();");
            let l = self.layouts[id].clone();
            let lc = log2(bn);
            self.used_tb.insert(id);
            let mut rows: Vec<u32> = Vec::new();
            let mut cols: Vec<u32> = Vec::new();
            let mut which = Vec::new();
            for k in 0..nr {
                let rp = l.reg_pos(k);
                let (r, cc) = (rp >> lc, rp & (bn as u32 - 1));
                let ri = rows.iter().position(|&x| x == r).unwrap_or_else(|| {
                    rows.push(r);
                    rows.len() - 1
                });
                let ci = cols.iter().position(|&x| x == cc).unwrap_or_else(|| {
                    cols.push(cc);
                    cols.len() - 1
                });
                which.push((ri, ci));
            }
            self.line("#pragma unroll 2");
            self.line(&format!("for (int kk = 0; kk < {bk}; kk++) {{"));
            self.ind += 1;
            self.line(&format!("float ar[{}], bc[{}];", rows.len(), cols.len()));
            for (i, r) in rows.iter().enumerate() {
                self.line(&format!("ar[{i}] = {sa}[kk * {lda} + ((tb{id} >> {lc}) ^ {r})];"));
            }
            for (i, cc) in cols.iter().enumerate() {
                self.line(&format!("bc[{i}] = {sb}[kk * {ldb} + ((tb{id} & {}) ^ {cc})];", bn - 1));
            }
            for (k, (ri, ci)) in which.iter().enumerate() {
                self.line(&format!("{acc}[{k}] = fmaf(ar[{ri}], bc[{ci}], {acc}[{k}]);"));
            }
            self.ind -= 1;
            self.line("}");
        }
        self.cache(out, id, acc);
    }

    /// Bytes of shared memory for a matmul's staging, at a fresh offset.
    fn alloc(&mut self, bytes: usize) -> usize {
        let off = self.dot_smem;
        self.dot_smem += bytes.next_multiple_of(16);
        off
    }

    /// A matmul's A operand straight from registers: a tile whose layout is
    /// the accumulator layout of the same rows with every warp on its own
    /// rows (as P in attention, the result of the previous matmul).
    fn reg_operand(&mut self, a: V, out: usize) -> Option<String> {
        let t = self.ty(a);
        let Lay::L(la) = self.lay[a] else { return None };
        let w = self.o.warps;
        let want = Layout::mma(&t.shape, w, 1);
        let rowwise_out = self.layouts[out] == Layout::mma(&self.layouts[out].shape.clone(), w, 1);
        if !rowwise_out || self.layouts[la] != want || self.raw.contains_key(&a) {
            return None;
        }
        Some(self.get(a, la))
    }

    /// Where a matmul operand sits in shared memory: (pointer, row length,
    /// transposed), staging it here unless it was staged before its loop.
    /// Inside a loop, staging alternates between two buffers by iteration,
    /// so one barrier per iteration suffices.
    fn place(&mut self, x: V, is_a: bool, staged: &mut bool) -> (String, usize, bool) {
        if let Some(p) = self.prestaged.get(&(x, is_a)) {
            return p.clone();
        }
        let (name, l, raw, tr) = self.operand(x);
        let t = self.ty(x);
        let (r, c) = if tr {
            (t.shape[1], t.shape[0])
        } else {
            (t.shape[0], t.shape[1])
        };
        let ld = c;
        let bytes = (r * ld * 2).next_multiple_of(16);
        let s = self.fresh("S");
        match self.loops.last().cloned() {
            Some(par) => {
                let off = self.alloc(2 * bytes);
                self.line(&format!(
                    "TSP khalf *{s} = (TSP khalf *)(tsmem + TSO_DOT + {off} + {par} * {bytes});"
                ));
            }
            None => {
                let off = self.alloc(bytes);
                self.line(&format!("TSP khalf *{s} = (TSP khalf *)(tsmem + TSO_DOT + {off});"));
            }
        }
        let v = if raw { Some(self.raw[&self.raw_src(x)]) } else { None };
        self.stage(&s, ld, &name, l, v);
        *staged = true;
        (s, ld, tr)
    }

    /// Before a loop: stages once the matmul operands its body uses but
    /// does not compute.
    fn prestage(&mut self, body: &'a [Inst]) {
        let defs = defs_in(body);
        let mut todo = Vec::new();
        for i in body {
            if let Op::Dot(a, b, _) = &i.op
                && matches!(self.lay[i.outs[0]], Lay::L(id) if self.mma.contains(&id))
            {
                let id = self.lid_of(i.outs[0]);
                for (x, is_a) in [(*a, true), (*b, false)] {
                    let computed_a = is_a && {
                        let w = self.o.warps;
                        matches!(self.lay[x], Lay::L(la) if self.layouts[la] == Layout::mma(&self.ty(x).shape, w, 1))
                            && self.layouts[id] == Layout::mma(&self.layouts[id].shape.clone(), w, 1)
                    };
                    if !defs.contains(&x) && !computed_a && !self.prestaged.contains_key(&(x, is_a)) {
                        todo.push((x, is_a));
                    }
                }
            }
        }
        if todo.is_empty() {
            return;
        }
        self.line("KSYNC();");
        let saved = std::mem::take(&mut self.loops);
        for (x, is_a) in todo {
            let mut staged = false;
            let p = self.place(x, is_a, &mut staged);
            self.prestaged.insert((x, is_a), p);
        }
        self.loops = saved;
        self.line("KSYNC();");
    }

    /// The packed load behind a matmul operand.
    fn raw_src(&self, x: V) -> V {
        match self.def[x] {
            Some(Inst { op: Op::Trans(y), .. }) if self.raw.contains_key(y) => *y,
            _ => x,
        }
    }

    // ---------------- software pipelining ----------------

    /// Tile loads at the top level of a loop body whose indices depend
    /// only on the loop index and values from outside the loop, from
    /// arrays the loop does not write.
    fn prefetchable(&self, body: &'a [Inst], iv: V, args: &[V]) -> Vec<&'a Inst> {
        let defs = defs_in(body);
        let written = stored_arrays(body);
        body.iter()
            .filter(|i| match &i.op {
                Op::Load { arr, idx, .. } => {
                    !self.ty(i.outs[0]).is_scalar()
                        && !written.contains(arr)
                        && idx.iter().all(|ix| {
                            let v = match ix {
                                Ix::Point(v) | Ix::Slice(v, _) => *v,
                            };
                            self.pure(v, iv, args, &defs)
                        })
                }
                _ => false,
            })
            .collect()
    }

    fn pure(&self, v: V, iv: V, args: &[V], defs: &std::collections::HashSet<V>) -> bool {
        if v == iv || !defs.contains(&v) {
            return !args.contains(&v);
        }
        let Some(i) = self.def[v] else { return false };
        match &i.op {
            Op::Const(_) | Op::ProgramId(_) | Op::ScalarArg(_) => true,
            Op::Unary(..) | Op::Binary(..) | Op::Where(..) | Op::Cast(..) => {
                self.ty(v).is_scalar() && Self::operands(&i.op).iter().all(|&x| self.pure(x, iv, args, defs))
            }
            Op::Load { idx, .. } => {
                self.ty(v).is_scalar()
                    && idx.iter().all(|ix| match ix {
                        Ix::Point(x) | Ix::Slice(x, _) => self.pure(*x, iv, args, defs),
                    })
            }
            _ => false,
        }
    }

    /// Index expressions of a load with the loop index replaced by `ive`.
    fn ix_subst(&self, idx: &[Ix], iv: V, ive: &str, body: &[Inst], _args: &[V]) -> Vec<(Option<usize>, String, V)> {
        let defs = defs_in(body);
        idx.iter()
            .map(|ix| match ix {
                Ix::Point(p) => (None, self.sexpr(*p, iv, ive, &defs), *p),
                Ix::Slice(s, n) => (Some(*n), self.sexpr(*s, iv, ive, &defs), *s),
            })
            .collect()
    }

    /// A scalar as one C expression, with the loop index replaced.
    fn sexpr(&self, v: V, iv: V, ive: &str, defs: &std::collections::HashSet<V>) -> String {
        if v == iv {
            return ive.to_string();
        }
        if !defs.contains(&v) {
            return self.sc(v);
        }
        let i = self.def[v].unwrap();
        let d = self.ty(v).dtype;
        match &i.op {
            Op::Const(x) => lit(*x, d),
            Op::ProgramId(a) => format!("(int)blockIdx.{}", ["x", "y", "z"][*a]),
            Op::ScalarArg(p) => format!("a{p}"),
            Op::Load { arr, idx, other } => {
                let PKind::Array { dims, dtype } = &self.k.params[*arr].kind else {
                    unreachable!()
                };
                let st = strides(dims);
                let mut off = Vec::new();
                let mut cond = Vec::new();
                for (k, ix) in idx.iter().enumerate() {
                    let Ix::Point(x) = ix else { unreachable!() };
                    let e = self.sexpr(*x, iv, ive, defs);
                    off.push(format!("(KI64)({e}) * {}", st[k]));
                    cond.push(format!("(unsigned)({e}) < {}u", dims[k]));
                }
                let p = format!("a{arr}[{}]", off.join(" + "));
                let rd = match dtype {
                    DType::F16 => format!("kh2f({p})"),
                    DType::Bool => format!("({p} != 0)"),
                    _ => p,
                };
                format!("(({}) ? {rd} : {})", cond.join(" && "), lit(*other, *dtype))
            }
            _ => {
                let args: Vec<String> = Self::operands(&i.op)
                    .iter()
                    .map(|&x| self.sexpr(x, iv, ive, defs))
                    .collect();
                self.expr_of(i, &args)
            }
        }
    }
}

/// Does some matmul take as A a tile that is neither a load nor a
/// transposed load (a computed tile, such as attention's probabilities)?
fn computed_a_operand(b: &[Inst]) -> bool {
    let mut defs: HashMap<V, &Op> = HashMap::new();
    fn walk<'b>(b: &'b [Inst], defs: &mut HashMap<V, &'b Op>, found: &mut bool) {
        for i in b {
            for &o in &i.outs {
                defs.insert(o, &i.op);
            }
            match &i.op {
                Op::Dot(a, _, _) => {
                    let is_load = |v: &V, d: &HashMap<V, &Op>| matches!(d.get(v), Some(Op::Load { .. }));
                    let ok = is_load(a, defs) || matches!(defs.get(a), Some(Op::Trans(y)) if is_load(y, defs));
                    if !ok {
                        *found = true;
                    }
                }
                Op::For { body, .. } => walk(body, defs, found),
                _ => {}
            }
        }
    }
    let mut found = false;
    walk(b, &mut defs, &mut found);
    found
}

fn defs_in(b: &[Inst]) -> std::collections::HashSet<V> {
    let mut s = std::collections::HashSet::new();
    for i in b {
        s.extend(i.outs.iter().copied());
        if let Op::For { body, iv, args, .. } = &i.op {
            s.insert(*iv);
            s.extend(args.iter().copied());
            s.extend(defs_in(body));
        }
    }
    s
}

/// Every value an operation reads.
fn all_operands(op: &Op) -> Vec<V> {
    match op {
        Op::Const(_) | Op::ProgramId(_) | Op::ScalarArg(_) | Op::Arange => vec![],
        Op::Splat(x)
        | Op::ExpandDims(x, _)
        | Op::Broadcast(x)
        | Op::Unary(_, x)
        | Op::Cast(x)
        | Op::Reduce(_, x, _)
        | Op::Trans(x) => {
            vec![*x]
        }
        Op::Binary(_, a, b) => vec![*a, *b],
        Op::Where(c, a, b) | Op::Dot(a, b, c) => vec![*a, *b, *c],
        Op::Load { idx, .. } => idx
            .iter()
            .map(|ix| match ix {
                Ix::Point(v) | Ix::Slice(v, _) => *v,
            })
            .collect(),
        Op::Store { idx, value, .. } => {
            let mut v: Vec<V> = idx
                .iter()
                .map(|ix| match ix {
                    Ix::Point(v) | Ix::Slice(v, _) => *v,
                })
                .collect();
            v.push(*value);
            v
        }
        Op::For {
            start,
            end,
            step,
            init,
            yields,
            ..
        } => {
            let mut v = vec![*start, *end, *step];
            v.extend(init);
            v.extend(yields);
            v
        }
    }
}

impl G<'_> {}

/// Generates the CUDA source of `k` (which also compiles as C++ against the
/// emulator in the prelude).
pub fn generate(k: &Kernel, o: &Options) -> Result<Generated, String> {
    let n = k.types.len();
    let mut g = G {
        k,
        o,
        layouts: Vec::new(),
        index: HashMap::new(),
        prio: Vec::new(),
        lay: vec![Lay::Unset; n],
        def: vec![None; n],
        div: vec![1; n],
        rng: vec![None; n],
        out: String::new(),
        ind: 1,
        scopes: vec![HashMap::new()],
        scal: HashMap::new(),
        n: 0,
        scratch: 0,
        dot_smem: 0,
        used_tb: BTreeSet::new(),
        mma: std::collections::HashSet::new(),
        raw: HashMap::new(),
        prefetched: std::collections::HashSet::new(),
        loops: Vec::new(),
        prestaged: HashMap::new(),
        rowwise: false,
    };
    g.rowwise = computed_a_operand(&k.body);
    g.assign(&k.body);
    g.divs(&k.body);
    g.rng = crate::analysis::ranges(k);
    g.find_raw();
    g.gen_block(&k.body);
    let threads = 32 * o.warps;
    let name = format!("tsl_{}", k.name);
    let scratch = g.scratch.next_multiple_of(16);
    let smem = scratch + g.dot_smem;
    let mut src = String::new();
    src.push_str(PRELUDE);
    let _ = writeln!(src, "\n#define TSO_DOT {scratch}");
    let mut params = Vec::new();
    let mut call = Vec::new();
    let stored = stored_arrays(&k.body);
    for (i, p) in k.params.iter().enumerate() {
        match &p.kind {
            PKind::Array { dtype, .. } => {
                let t = match dtype {
                    DType::F32 => "float",
                    DType::F16 => "khalf",
                    DType::I32 => "int",
                    DType::Bool => "unsigned char",
                };
                let c = if stored.contains(&i) { "" } else { "const " };
                params.push(format!("{c}{t} *__restrict__ a{i}"));
                call.push(format!("({c}{t} *)A[{i}]"));
            }
            PKind::Scalar(d) => {
                let t = cty(*d);
                params.push(format!("const {t} a{i}"));
                call.push(format!("*(const {t} *)A[{i}]"));
            }
        }
    }
    let _ = writeln!(src, "KGLOBAL({threads}) {name}({}) {{", params.join(", "));
    let scratch_ptrs =
        "  TSP float *tsf = (TSP float *)tsmem;\n  TSP int *tsi = (TSP int *)tsmem;\n  (void)tsf;\n  (void)tsi;\n";
    src.push_str("  SMEM;\n");
    src.push_str(scratch_ptrs);
    let mut body = String::new();
    body.push_str("  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  (void)g;\n  (void)t;\n  (void)warp;\n");
    for &id in &g.used_tb {
        let l = &g.layouts[id];
        let mut terms = Vec::new();
        for i in 0..5 {
            if l.lane[i] != 0 {
                terms.push(format!("(((lane >> {i}) & 1) * {})", l.lane[i]));
            }
        }
        for (i, b) in l.warp.iter().enumerate() {
            if *b != 0 {
                terms.push(format!("(((warp >> {i}) & 1) * {b})"));
            }
        }
        let e = if terms.is_empty() {
            "0".into()
        } else {
            terms.join(" ^ ")
        };
        let _ = writeln!(body, "  const int tb{id} = {e};");
    }
    body.push_str(&g.out);
    src.push_str(&body);
    src.push_str("}\n");
    let _ = writeln!(
        src,
        "#ifndef TSL_CUDA\nstatic void {name}_body(float **A) {{ {name}({}); }}\nextern \"C\" void {name}_emu(float **A, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned smem) {{ kemu_launch({name}_body, A, gx, gy, gz, bx, 1, smem); }}\n#endif",
        call.join(", ")
    );
    let metal = if o.arch == METAL {
        if smem > 32 * 1024 {
            return Err(format!(
                "{}: needs {} KB of threadgroup memory; Metal allows 32",
                k.name,
                smem.div_ceil(1024)
            ));
        }
        let mut m = String::from(METAL_PRELUDE);
        let _ = writeln!(m, "\n#define TSO_DOT {scratch}");
        let mut ps = Vec::new();
        for (i, p) in k.params.iter().enumerate() {
            match &p.kind {
                PKind::Array { dtype, .. } => {
                    let t = match dtype {
                        DType::F32 => "float",
                        DType::F16 => "khalf",
                        DType::I32 => "int",
                        DType::Bool => "uchar",
                    };
                    let c = if stored.contains(&i) { "" } else { "const " };
                    ps.push(format!("device {c}{t} *a{i} [[buffer({i})]]"));
                }
                PKind::Scalar(d) => ps.push(format!("constant {} &a{i} [[buffer({i})]]", cty(*d))),
            }
        }
        ps.push("uint3 blockIdx [[threadgroup_position_in_grid]]".into());
        ps.push("uint3 threadIdx [[thread_position_in_threadgroup]]".into());
        let _ = writeln!(m, "kernel void {name}({}) {{", ps.join(", "));
        let _ = writeln!(
            m,
            "  threadgroup float4 tsmem4[{}];\n  threadgroup uchar *tsmem = (threadgroup uchar *)tsmem4;",
            smem.div_ceil(16).max(1)
        );
        m.push_str(scratch_ptrs);
        m.push_str(&body);
        m.push_str("}\n");
        Some(m)
    } else {
        None
    };
    Ok(Generated {
        name,
        metal,
        source: src,
        threads,
        smem,
        grid: k.grid,
    })
}

fn stored_arrays(b: &[Inst]) -> Vec<usize> {
    let mut out = Vec::new();
    for i in b {
        match &i.op {
            Op::Store { arr, .. } => out.push(*arr),
            Op::For { body, .. } => out.extend(stored_arrays(body)),
            _ => {}
        }
    }
    out
}
