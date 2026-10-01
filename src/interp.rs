//! A reference interpreter for the tile IR: runs every program of the grid
//! one after another, on the CPU, with the semantics every backend must
//! reproduce. It is the oracle the generated code is tested against.

use crate::ast::DType;
use crate::half::round_f16;
use crate::ir::{Bin, Inst, Ix, Kernel, Op, PKind, Red, Un, V};

/// A host array or scalar argument. Values are kept as f32 whatever the
/// dtype: f16 arrays hold values representable in f16, i32 arrays small
/// integers (exact below 2^24), bool arrays 0 and 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(dtype: DType, shape: &[usize], data: Vec<f32>) -> Tensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "data does not match shape {shape:?}"
        );
        let data = data.into_iter().map(|x| round(dtype, x as f64) as f32).collect();
        Tensor {
            dtype,
            shape: shape.to_vec(),
            data,
        }
    }
    pub fn zeros(dtype: DType, shape: &[usize]) -> Tensor {
        Tensor {
            dtype,
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product()],
        }
    }
    pub fn scalar(dtype: DType, x: f32) -> Tensor {
        Tensor::new(dtype, &[], vec![x])
    }
}

/// `x` as stored in `dtype`.
pub fn round(dtype: DType, x: f64) -> f64 {
    match dtype {
        DType::F32 => x as f32 as f64,
        DType::F16 => round_f16(x as f32) as f64,
        DType::I32 => {
            if x.is_nan() {
                0.0
            } else {
                (x.trunc() as i64 as i32) as f64
            }
        }
        DType::Bool => (x != 0.0) as u8 as f64,
    }
}

struct Run<'a> {
    k: &'a Kernel,
    args: &'a mut [Tensor],
    vals: Vec<Vec<f64>>,
    pid: [usize; 3],
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

impl Run<'_> {
    fn shape(&self, v: V) -> &[usize] {
        &self.k.types[v].shape
    }
    fn dt(&self, v: V) -> DType {
        self.k.types[v].dtype
    }
    fn set(&mut self, v: V, data: Vec<f64>) {
        let dt = self.dt(v);
        self.vals[v] = data.into_iter().map(|x| round(dt, x)).collect();
    }
    fn scalar(&self, v: V) -> f64 {
        self.vals[v][0]
    }

    /// The array positions a block covers, in row-major tile order: the
    /// linear offset of each element, or None outside the array.
    fn block(&self, arr: usize, idx: &[Ix]) -> Vec<Option<usize>> {
        let PKind::Array { dims, .. } = &self.k.params[arr].kind else {
            unreachable!()
        };
        let st = strides(dims);
        let mut out = vec![Some(0usize)];
        for (d, ix) in idx.iter().enumerate() {
            let (start, n) = match ix {
                Ix::Point(v) => (self.scalar(*v) as i64, 1),
                Ix::Slice(v, n) => (self.scalar(*v) as i64, *n),
            };
            let mut next = Vec::with_capacity(out.len() * n);
            for o in &out {
                for j in 0..n {
                    let p = start + j as i64;
                    next.push(match o {
                        Some(o) if p >= 0 && (p as usize) < dims[d] => Some(o + p as usize * st[d]),
                        _ => None,
                    });
                }
            }
            out = next;
        }
        out
    }

    fn block_of(&mut self, b: &[Inst]) {
        for i in b {
            self.inst(i);
        }
    }

    fn inst(&mut self, i: &Inst) {
        let out = i.outs.first().copied();
        match &i.op {
            Op::Const(x) => self.set(out.unwrap(), vec![*x]),
            Op::ProgramId(a) => self.set(out.unwrap(), vec![self.pid[*a] as f64]),
            Op::ScalarArg(p) => {
                let x = self.args[*p].data[0] as f64;
                self.set(out.unwrap(), vec![x]);
            }
            Op::Arange => {
                let n = self.shape(out.unwrap())[0];
                self.set(out.unwrap(), (0..n).map(|x| x as f64).collect());
            }
            Op::Splat(v) => {
                let o = out.unwrap();
                let n: usize = self.shape(o).iter().product();
                let x = self.scalar(*v);
                self.set(o, vec![x; n]);
            }
            Op::ExpandDims(v, _) => {
                let d = self.vals[*v].clone();
                self.set(out.unwrap(), d);
            }
            Op::Broadcast(v) => {
                let o = out.unwrap();
                let (src, dst) = (self.shape(*v).to_vec(), self.shape(o).to_vec());
                let ss = strides(&src);
                let n: usize = dst.iter().product();
                let ds = strides(&dst);
                let data = (0..n)
                    .map(|e| {
                        let mut off = 0;
                        for d in 0..dst.len() {
                            let p = (e / ds[d]) % dst[d];
                            if src[d] != 1 {
                                off += p * ss[d];
                            }
                        }
                        self.vals[*v][off]
                    })
                    .collect();
                self.set(o, data);
            }
            Op::Unary(op, v) => {
                let d = self.vals[*v]
                    .iter()
                    .map(|&x| match op {
                        Un::Neg => -x,
                        Un::Exp => (x as f32).exp() as f64,
                        Un::Log => (x as f32).ln() as f64,
                        Un::Sqrt => (x as f32).sqrt() as f64,
                        Un::Rsqrt => 1.0 / (x as f32).sqrt() as f64,
                        Un::Abs => x.abs(),
                        Un::Not => (x == 0.0) as u8 as f64,
                    })
                    .collect();
                self.set(out.unwrap(), d);
            }
            Op::Binary(op, a, b) => {
                let int = !self.dt(*a).is_float();
                let d = self.vals[*a]
                    .iter()
                    .zip(&self.vals[*b])
                    .map(|(&x, &y)| match op {
                        Bin::Add => x + y,
                        Bin::Sub => x - y,
                        Bin::Mul => x * y,
                        Bin::Div => x / y,
                        Bin::FloorDiv => {
                            if y == 0.0 {
                                0.0
                            } else {
                                (x / y).floor()
                            }
                        }
                        Bin::Mod => {
                            if y == 0.0 {
                                0.0
                            } else {
                                x - y * (x / y).floor()
                            }
                        }
                        Bin::Max => {
                            if x >= y || y.is_nan() {
                                x
                            } else {
                                y
                            }
                        }
                        Bin::Min => {
                            if x <= y || y.is_nan() {
                                x
                            } else {
                                y
                            }
                        }
                        Bin::Lt => (x < y) as u8 as f64,
                        Bin::Le => (x <= y) as u8 as f64,
                        Bin::Gt => (x > y) as u8 as f64,
                        Bin::Ge => (x >= y) as u8 as f64,
                        Bin::Eq => (x == y) as u8 as f64,
                        Bin::Ne => (x != y) as u8 as f64,
                        Bin::And => (x != 0.0 && y != 0.0) as u8 as f64,
                        Bin::Or => (x != 0.0 || y != 0.0) as u8 as f64,
                    })
                    .collect::<Vec<f64>>();
                // f32 arithmetic happens in f32.
                let _ = int;
                self.set(out.unwrap(), d);
            }
            Op::Where(c, a, b) => {
                let d = (0..self.vals[*c].len())
                    .map(|e| {
                        if self.vals[*c][e] != 0.0 {
                            self.vals[*a][e]
                        } else {
                            self.vals[*b][e]
                        }
                    })
                    .collect();
                self.set(out.unwrap(), d);
            }
            Op::Cast(v) => {
                let d = self.vals[*v].clone();
                self.set(out.unwrap(), d);
            }
            Op::Reduce(op, v, axis) => {
                let s = self.shape(*v).to_vec();
                let (outer, n, inner) = if s.len() == 1 {
                    (1, s[0], 1)
                } else if *axis == 0 {
                    (1, s[0], s[1])
                } else {
                    (s[0], s[1], 1)
                };
                let x = &self.vals[*v];
                let mut d = Vec::new();
                for o in 0..outer {
                    for i in 0..inner {
                        let it = (0..n).map(|j| x[o * n * inner + j * inner + i]);
                        d.push(match op {
                            Red::Sum => it.sum::<f64>(),
                            Red::Max => it.fold(f64::NEG_INFINITY, f64::max),
                            Red::Min => it.fold(f64::INFINITY, f64::min),
                        });
                    }
                }
                self.set(out.unwrap(), d);
            }
            Op::Trans(v) => {
                let s = self.shape(*v).to_vec();
                let x = &self.vals[*v];
                let d = (0..s[0] * s[1]).map(|e| x[(e % s[0]) * s[1] + e / s[0]]).collect();
                self.set(out.unwrap(), d);
            }
            Op::Dot(a, b, c) => {
                let (m, k) = (self.shape(*a)[0], self.shape(*a)[1]);
                let n = self.shape(*b)[1];
                let (x, y, z) = (&self.vals[*a], &self.vals[*b], &self.vals[*c]);
                let mut d = vec![0.0; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0.0;
                        for l in 0..k {
                            s += x[i * k + l] * y[l * n + j];
                        }
                        d[i * n + j] = z[i * n + j] + s;
                    }
                }
                self.set(out.unwrap(), d);
            }
            Op::Load { arr, idx, other } => {
                let offs = self.block(*arr, idx);
                let a = &self.args[*arr];
                let d = offs
                    .iter()
                    .map(|o| o.map(|o| a.data[o] as f64).unwrap_or(*other))
                    .collect();
                self.set(out.unwrap(), d);
            }
            Op::Store { arr, idx, value } => {
                let offs = self.block(*arr, idx);
                let dt = self.args[*arr].dtype;
                for (e, o) in offs.iter().enumerate() {
                    if let Some(o) = o {
                        self.args[*arr].data[*o] = round(dt, self.vals[*value][e]) as f32;
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
                for (a, x) in args.iter().zip(init) {
                    self.vals[*a] = self.vals[*x].clone();
                }
                let (s, e, st) = (
                    self.scalar(*start) as i64,
                    self.scalar(*end) as i64,
                    self.scalar(*step) as i64,
                );
                let mut t = s;
                while t < e {
                    self.vals[*iv] = vec![t as f64];
                    self.block_of(body);
                    let ys: Vec<Vec<f64>> = yields.iter().map(|y| self.vals[*y].clone()).collect();
                    for (a, y) in args.iter().zip(ys) {
                        self.vals[*a] = y;
                    }
                    t += st;
                }
                for (o, a) in i.outs.iter().zip(args) {
                    self.vals[*o] = self.vals[*a].clone();
                }
            }
        }
    }
}

/// Runs `k` over its whole grid. `args` are the kernel's parameters in
/// order: arrays of the specialized shapes, and scalars.
pub fn run(k: &Kernel, args: &mut [Tensor]) -> Result<(), String> {
    check_args(k, args)?;
    let mut r = Run {
        k,
        args,
        vals: vec![Vec::new(); k.types.len()],
        pid: [0; 3],
    };
    for z in 0..k.grid[2] {
        for y in 0..k.grid[1] {
            for x in 0..k.grid[0] {
                r.pid = [x, y, z];
                r.block_of(&k.body);
            }
        }
    }
    Ok(())
}

pub fn check_args(k: &Kernel, args: &[Tensor]) -> Result<(), String> {
    if args.len() != k.params.len() {
        return Err(format!(
            "{} takes {} arguments, {} given",
            k.name,
            k.params.len(),
            args.len()
        ));
    }
    for (p, a) in k.params.iter().zip(args) {
        match &p.kind {
            PKind::Array { dtype, dims } => {
                if a.dtype != *dtype || a.shape != *dims {
                    return Err(format!(
                        "{}: {} is {}{:?}, given {}{:?}",
                        k.name,
                        p.name,
                        dtype.name(),
                        dims,
                        a.dtype.name(),
                        a.shape
                    ));
                }
            }
            PKind::Scalar(dt) => {
                if a.dtype != *dt || !a.shape.is_empty() {
                    return Err(format!("{}: {} is a {} scalar", k.name, p.name, dt.name()));
                }
            }
        }
    }
    Ok(())
}
