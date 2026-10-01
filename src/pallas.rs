//! TPU code generation: the tile IR as a Pallas kernel (JAX's kernel
//! language for TPUs), emitted as a Python module.
//!
//! Arrays the kernel reads or writes in blocks stay in HBM; each block
//! access becomes a DMA between HBM and a VMEM buffer of the tile's shape.
//! Arrays read only one element at a time (block tables, lengths) and
//! scalar arguments live in SMEM. Tiles are JAX arrays, so elementwise
//! operations, reductions and `dot` (on the MXU, with f32 accumulation)
//! map directly; loops become `fori_loop` with the carried tiles.
//!
//! TPUs compute in bfloat16, not float16: the kernel's f16 tiles and
//! arrays are bf16 here. DMAs must stay inside their arrays, so every
//! block access must be provably in bounds (shapes that are multiples of
//! the block sizes); the generator refuses kernels it cannot prove.

use crate::analysis::{in_bounds, ranges};
use crate::ast::DType;
use crate::ir::{Bin, Inst, Ix, Kernel, Op, PKind, Red, Un, V};
use std::fmt::Write;

fn jdt(d: DType) -> &'static str {
    match d {
        DType::F32 => "jnp.float32",
        DType::F16 => "jnp.bfloat16",
        DType::I32 => "jnp.int32",
        DType::Bool => "jnp.bool_",
    }
}

fn lit(x: f64, d: DType) -> String {
    match d {
        DType::I32 => format!("jnp.int32({})", x as i64),
        DType::Bool => format!("jnp.bool_({})", if x != 0.0 { "True" } else { "False" }),
        _ => {
            let v = if x.is_nan() {
                "float('nan')".to_string()
            } else if x.is_infinite() {
                if x > 0.0 {
                    "float('inf')".into()
                } else {
                    "float('-inf')".into()
                }
            } else {
                format!("{x:?}")
            };
            format!("{}({v})", jdt(d))
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mem {
    /// Read and written in blocks, by DMA.
    Hbm,
    /// Read one element at a time.
    Smem,
    Unused,
}

struct P<'a> {
    k: &'a Kernel,
    rng: Vec<Option<(i64, i64)>>,
    mem: Vec<Mem>,
    /// Arrays the kernel writes, in output order.
    outs: Vec<usize>,
    /// VMEM buffers: (shape, dtype).
    bufs: Vec<(Vec<usize>, DType)>,
    out: String,
    ind: usize,
    n: usize,
}

fn walk<'b>(b: &'b [Inst], f: &mut dyn FnMut(&'b Inst)) {
    for i in b {
        f(i);
        if let Op::For { body, .. } = &i.op {
            walk(body, f);
        }
    }
}

impl P<'_> {
    fn line(&mut self, s: &str) {
        for _ in 0..self.ind {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn v(&self, v: V) -> String {
        format!("v{v}")
    }
    fn fresh(&mut self) -> usize {
        self.n += 1;
        self.n
    }

    /// The array reference a block access names: writes and reads of an
    /// array the kernel writes go through its output reference.
    fn aref(&self, arr: usize) -> String {
        match self.outs.iter().position(|&o| o == arr) {
            Some(i) => format!("o{i}"),
            None => format!("a{arr}"),
        }
    }

    /// The reference to the block `idx` of array `arr`, and the condition
    /// under which it lies inside the array. Slices must be provably in
    /// bounds; a point index that may not be (one computed from loaded
    /// data, like a page number) is clamped into the array, and the access
    /// is masked by the condition instead.
    fn view(&mut self, arr: usize, idx: &[Ix]) -> Result<(String, Option<String>), String> {
        let PKind::Array { dims, .. } = &self.k.params[arr].kind else {
            unreachable!()
        };
        let dims = dims.clone();
        let (mut parts, mut conds) = (Vec::new(), Vec::new());
        for (d, ix) in idx.iter().enumerate() {
            match ix {
                Ix::Point(p) if in_bounds(&self.rng, *p, 1, dims[d]) => parts.push(self.v(*p)),
                Ix::Point(p) => {
                    let (v, n) = (self.v(*p), dims[d]);
                    let c = format!("c{}", self.fresh());
                    self.line(&format!("{c} = jnp.clip({v}, 0, {})", n - 1));
                    conds.push(format!("({v} >= 0) & ({v} < {n})"));
                    parts.push(c);
                }
                Ix::Slice(s, n) => {
                    if !in_bounds(&self.rng, *s, *n, dims[d]) {
                        return Err(self.oob(arr));
                    }
                    parts.push(format!("pl.ds({}, {n})", self.v(*s)));
                }
            }
        }
        let cond = if conds.is_empty() {
            None
        } else {
            Some(conds.join(" & "))
        };
        Ok((format!("{}.at[{}]", self.aref(arr), parts.join(", ")), cond))
    }

    fn oob(&self, arr: usize) -> String {
        format!(
            "{}: the TPU backend needs every block access provably inside its array (a block of {} may run past its end; use shapes that are multiples of the block sizes)",
            self.k.name, self.k.params[arr].name
        )
    }

    fn block(&mut self, b: &[Inst]) -> Result<(), String> {
        for i in b {
            self.inst(i)?;
        }
        Ok(())
    }

    fn inst(&mut self, i: &Inst) -> Result<(), String> {
        let o = i.outs.first().map(|&o| self.v(o)).unwrap_or_default();
        let ty = |v: V| &self.k.types[v];
        match &i.op {
            Op::Const(x) => {
                let e = lit(*x, ty(i.outs[0]).dtype);
                self.line(&format!("{o} = {e}"));
            }
            Op::ProgramId(a) => self.line(&format!("{o} = pl.program_id({a})")),
            Op::ScalarArg(p) => self.line(&format!("{o} = a{p}[0]")),
            Op::Arange => {
                let n = ty(i.outs[0]).shape[0];
                self.line(&format!("{o} = jax.lax.broadcasted_iota(jnp.int32, ({n},), 0)"));
            }
            Op::Splat(x) => {
                let t = ty(i.outs[0]);
                self.line(&format!(
                    "{o} = jnp.full({}, {}, {})",
                    tup(&t.shape),
                    self.v(*x),
                    jdt(t.dtype)
                ));
            }
            Op::ExpandDims(x, a) => self.line(&format!("{o} = jnp.expand_dims({}, {a})", self.v(*x))),
            Op::Broadcast(x) => {
                let t = ty(i.outs[0]);
                self.line(&format!("{o} = jnp.broadcast_to({}, {})", self.v(*x), tup(&t.shape)));
            }
            Op::Unary(op, x) => {
                let x = self.v(*x);
                let e = match op {
                    Un::Neg => format!("-{x}"),
                    Un::Exp => format!("jnp.exp({x})"),
                    Un::Log => format!("jnp.log({x})"),
                    Un::Sqrt => format!("jnp.sqrt({x})"),
                    Un::Rsqrt => format!("jax.lax.rsqrt({x})"),
                    Un::Abs => format!("jnp.abs({x})"),
                    Un::Not => format!("jnp.logical_not({x})"),
                };
                self.line(&format!("{o} = {e}"));
            }
            Op::Binary(op, a, b) => {
                let (a, b) = (self.v(*a), self.v(*b));
                let e = match op {
                    Bin::Add => format!("{a} + {b}"),
                    Bin::Sub => format!("{a} - {b}"),
                    Bin::Mul => format!("{a} * {b}"),
                    Bin::Div => format!("{a} / {b}"),
                    Bin::FloorDiv => format!("jnp.floor_divide({a}, {b})"),
                    Bin::Mod => format!("jnp.remainder({a}, {b})"),
                    Bin::Max => format!("jnp.maximum({a}, {b})"),
                    Bin::Min => format!("jnp.minimum({a}, {b})"),
                    Bin::Lt => format!("{a} < {b}"),
                    Bin::Le => format!("{a} <= {b}"),
                    Bin::Gt => format!("{a} > {b}"),
                    Bin::Ge => format!("{a} >= {b}"),
                    Bin::Eq => format!("{a} == {b}"),
                    Bin::Ne => format!("{a} != {b}"),
                    Bin::And => format!("jnp.logical_and({a}, {b})"),
                    Bin::Or => format!("jnp.logical_or({a}, {b})"),
                };
                self.line(&format!("{o} = {e}"));
            }
            Op::Where(c, a, b) => self.line(&format!(
                "{o} = jnp.where({}, {}, {})",
                self.v(*c),
                self.v(*a),
                self.v(*b)
            )),
            Op::Cast(x) => {
                let d = ty(i.outs[0]).dtype;
                self.line(&format!("{o} = {}.astype({})", self.v(*x), jdt(d)));
            }
            Op::Reduce(op, x, a) => {
                let f = match op {
                    Red::Sum => "jnp.sum",
                    Red::Max => "jnp.max",
                    Red::Min => "jnp.min",
                };
                self.line(&format!("{o} = {f}({}, axis={a})", self.v(*x)));
            }
            Op::Trans(x) => self.line(&format!("{o} = {}.T", self.v(*x))),
            Op::Dot(a, b, c) => self.line(&format!(
                "{o} = {} + jnp.dot({}, {}, preferred_element_type=jnp.float32)",
                self.v(*c),
                self.v(*a),
                self.v(*b)
            )),
            Op::Load { arr, idx, other } => {
                let t = ty(i.outs[0]).clone();
                let (view, cond) = self.view(*arr, idx)?;
                let val = if t.is_scalar() {
                    // An SMEM array: index it directly.
                    view.replacen(".at[", "[", 1)
                } else {
                    self.bufs.push((t.shape.clone(), t.dtype));
                    let b = self.bufs.len() - 1;
                    self.line(&format!("pltpu.sync_copy({view}, b{b})"));
                    format!("b{b}[...]")
                };
                match cond {
                    None => self.line(&format!("{o} = {val}")),
                    Some(c) => self.line(&format!("{o} = jnp.where({c}, {val}, {})", lit(*other, t.dtype))),
                }
            }
            Op::Store { arr, idx, value } => {
                let t = ty(*value).clone();
                if t.is_scalar() {
                    return Err(format!(
                        "{}: the TPU backend does not store single elements",
                        self.k.name
                    ));
                }
                let (view, cond) = self.view(*arr, idx)?;
                self.bufs.push((t.shape.clone(), t.dtype));
                let b = self.bufs.len() - 1;
                self.line(&format!("b{b}[...] = {}", self.v(*value)));
                let copy = format!("pltpu.sync_copy(b{b}, {view})");
                match cond {
                    None => self.line(&copy),
                    Some(c) => {
                        let f = self.fresh();
                        self.line(&format!("@pl.when({c})"));
                        self.line(&format!("def _store{f}():"));
                        self.line(&format!("    {copy}"));
                    }
                }
            }
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
                let f = self.fresh();
                let names = |v: &[V]| v.iter().map(|x| format!("v{x}")).collect::<Vec<_>>();
                self.line(&format!("def body{f}(t, carry):"));
                self.ind += 1;
                if !args.is_empty() {
                    self.line(&format!("({},) = carry", names(args).join(", ")));
                }
                self.line(&format!("{} = {} + t * {}", self.v(*iv), self.v(*start), self.v(*step)));
                self.block(body)?;
                self.line(&format!(
                    "return ({}{})",
                    names(yields).join(", "),
                    if yields.is_empty() { "" } else { "," }
                ));
                self.ind -= 1;
                let (s, e, st) = (self.v(*start), self.v(*end), self.v(*step));
                self.line(&format!("n{f} = jnp.maximum(0, ({e} - {s} + {st} - 1) // {st})"));
                let res = format!(
                    "jax.lax.fori_loop(0, n{f}, body{f}, ({}{}))",
                    names(init).join(", "),
                    if init.is_empty() { "" } else { "," }
                );
                if i.outs.is_empty() {
                    self.line(&res);
                } else {
                    self.line(&format!("({},) = {res}", names(&i.outs).join(", ")));
                }
            }
        }
        Ok(())
    }
}

/// A shape as a Python tuple.
fn tup(s: &[usize]) -> String {
    match s {
        [x] => format!("({x},)"),
        _ => format!("({})", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

/// Python source defining `run(*args, interpret=False)`: the kernel over
/// its grid, taking the parameters in order (arrays and Python numbers)
/// and returning the arrays it writes, in parameter order.
pub fn generate(k: &Kernel) -> Result<String, String> {
    let mut mem = vec![Mem::Unused; k.params.len()];
    let mut outs = Vec::new();
    let mut err = None;
    walk(&k.body, &mut |i| match &i.op {
        Op::Load { arr, idx, .. } => {
            let scalar = idx.iter().all(|x| matches!(x, Ix::Point(_)));
            let m = if scalar { Mem::Smem } else { Mem::Hbm };
            if mem[*arr] != Mem::Unused && mem[*arr] != m {
                err = Some(format!(
                    "{}: the TPU backend needs {} read either in blocks or by element, not both",
                    k.name, k.params[*arr].name
                ));
            }
            mem[*arr] = m;
        }
        Op::Store { arr, .. } => {
            if mem[*arr] == Mem::Smem {
                err = Some(format!(
                    "{}: {} is both read by element and written",
                    k.name, k.params[*arr].name
                ));
            }
            mem[*arr] = Mem::Hbm;
            if !outs.contains(arr) {
                outs.push(*arr);
            }
        }
        _ => {}
    });
    if let Some(e) = err {
        return Err(e);
    }
    outs.sort();
    let mut p = P {
        k,
        rng: ranges(k),
        mem,
        outs,
        bufs: Vec::new(),
        out: String::new(),
        ind: 1,
        n: 0,
    };
    p.block(&k.body)?;
    let body = std::mem::take(&mut p.out);
    let np = k.params.len();
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# Generated by tessel from kernel {}: the TPU (Pallas) backend.",
        k.name
    );
    s.push_str("import jax\nimport jax.numpy as jnp\nfrom jax.experimental import pallas as pl\nfrom jax.experimental.pallas import tpu as pltpu\n\n");
    let refs: Vec<String> = (0..np)
        .map(|i| format!("a{i}"))
        .chain((0..p.outs.len()).map(|i| format!("o{i}")))
        .chain((0..p.bufs.len()).map(|i| format!("b{i}")))
        .collect();
    let _ = writeln!(s, "def _kernel({}):", refs.join(", "));
    s.push_str(&body);
    s.push_str("    return None\n\n");
    let _ = writeln!(s, "GRID = ({}, {}, {})", k.grid[0], k.grid[1], k.grid[2]);
    let _ = writeln!(s, "OUTPUTS = {:?}", p.outs);
    s.push_str("\ndef run(*args, interpret=False):\n");
    s.push_str("    ins, specs = [], []\n");
    for (i, prm) in k.params.iter().enumerate() {
        match &prm.kind {
            PKind::Array { dtype, dims } => {
                let _ = writeln!(
                    s,
                    "    ins.append(jnp.asarray(args[{i}], {}).reshape({}))",
                    jdt(*dtype),
                    tup(dims)
                );
                let space = match p.mem[i] {
                    Mem::Smem => "pltpu.SMEM",
                    _ => "pl.ANY",
                };
                let _ = writeln!(s, "    specs.append(pl.BlockSpec(memory_space={space}))");
            }
            PKind::Scalar(d) => {
                let _ = writeln!(s, "    ins.append(jnp.asarray([args[{i}]], {}))", jdt(*d));
                s.push_str("    specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))\n");
            }
        }
    }
    let shapes: Vec<String> = p
        .outs
        .iter()
        .map(|&a| match &k.params[a].kind {
            PKind::Array { dtype, dims } => format!("jax.ShapeDtypeStruct({}, {})", tup(dims), jdt(*dtype)),
            _ => unreachable!(),
        })
        .collect();
    let scratch: Vec<String> = p
        .bufs
        .iter()
        .map(|(sh, d)| format!("pltpu.VMEM({}, {})", tup(sh), jdt(*d)))
        .collect();
    let aliases: Vec<String> = p.outs.iter().enumerate().map(|(o, a)| format!("{a}: {o}")).collect();
    let grid = tup(&k.grid);
    let _ = writeln!(
        s,
        "    call = pl.pallas_call(\n        _kernel,\n        out_shape=[{}],\n        grid={grid},\n        in_specs=specs,\n        out_specs=[{}],\n        scratch_shapes=[{}],\n        input_output_aliases={{{}}},\n        interpret=pltpu.InterpretParams() if interpret else False,\n    )\n    return call(*ins)",
        shapes.join(", "),
        vec!["pl.BlockSpec(memory_space=pl.ANY)"; p.outs.len()].join(", "),
        scratch.join(", "),
        aliases.join(", ")
    );
    Ok(s)
}
