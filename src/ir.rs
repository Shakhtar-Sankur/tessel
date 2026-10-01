//! The tile IR: a kernel specialized for concrete array shapes and meta
//! constants, in SSA form. Every value has a static type: a scalar or a
//! tile of rank 1 or 2 with power-of-two dimensions, and a dtype.
//! Broadcasting is explicit (`Splat`, `ExpandDims`, `Broadcast`), so every
//! elementwise operation sees operands of one shape, and every compile-time
//! integer (array dimensions, meta constants and arithmetic on them) is
//! folded away.

use crate::ast::{self, BinOp as AB, DType, DimSpec, Expr, Index, KernelDef, ParamTy, Stmt};
use std::collections::HashMap;

pub type V = usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ty {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Ty {
    pub fn scalar(dtype: DType) -> Ty {
        Ty { shape: vec![], dtype }
    }
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Un {
    Neg,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Abs,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Max,
    Min,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl Bin {
    pub fn is_compare(self) -> bool {
        matches!(self, Bin::Lt | Bin::Le | Bin::Gt | Bin::Ge | Bin::Eq | Bin::Ne)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Red {
    Sum,
    Max,
    Min,
}

#[derive(Clone, Debug)]
pub enum Ix {
    /// One position along this dimension (the dimension disappears).
    Point(V),
    /// `size` positions from `start`.
    Slice(V, usize),
}

#[derive(Clone, Debug)]
pub enum Op {
    Const(f64),
    ProgramId(usize),
    ScalarArg(usize),
    Arange,
    Splat(V),
    ExpandDims(V, usize),
    Broadcast(V),
    Unary(Un, V),
    Binary(Bin, V, V),
    Where(V, V, V),
    Cast(V),
    Reduce(Red, V, usize),
    Trans(V),
    Dot(V, V, V),
    Load {
        arr: usize,
        idx: Vec<Ix>,
        other: f64,
    },
    Store {
        arr: usize,
        idx: Vec<Ix>,
        value: V,
    },
    For {
        iv: V,
        start: V,
        end: V,
        step: V,
        args: Vec<V>,
        init: Vec<V>,
        body: Vec<Inst>,
        yields: Vec<V>,
    },
}

#[derive(Clone, Debug)]
pub struct Inst {
    pub outs: Vec<V>,
    pub op: Op,
}

#[derive(Clone, Debug)]
pub enum PKind {
    Array { dtype: DType, dims: Vec<usize> },
    Scalar(DType),
}

#[derive(Clone, Debug)]
pub struct IrParam {
    pub name: String,
    pub kind: PKind,
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<IrParam>,
    pub meta: Vec<(String, i64)>,
    pub grid: [usize; 3],
    pub types: Vec<Ty>,
    pub body: Vec<Inst>,
}

impl Kernel {
    pub fn ty(&self, v: V) -> &Ty {
        &self.types[v]
    }
    pub fn meta(&self, name: &str) -> Option<i64> {
        self.meta.iter().find(|(n, _)| n == name).map(|m| m.1)
    }
}

/// The value of an expression while building: compile-time constants stay
/// symbolic until an operation needs them at run time.
#[derive(Clone, Debug)]
enum E {
    Int(i64),
    Float(f64),
    Val(V),
    Arr(usize),
    Shape(Vec<usize>),
    DT(DType),
}

struct B<'a> {
    def: &'a KernelDef,
    params: Vec<IrParam>,
    types: Vec<Ty>,
    blocks: Vec<Vec<Inst>>,
    env: HashMap<String, E>,
    meta: Vec<(String, i64)>,
    grid: Option<[usize; 3]>,
    consts: HashMap<(u64, DType), V>,
}

fn err<T>(line: usize, msg: impl std::fmt::Display) -> Result<T, String> {
    Err(format!("line {line}: {msg}"))
}

impl B<'_> {
    fn new_val(&mut self, ty: Ty) -> V {
        self.types.push(ty);
        self.types.len() - 1
    }
    fn emit(&mut self, op: Op, ty: Ty) -> V {
        let v = self.new_val(ty);
        self.blocks.last_mut().unwrap().push(Inst { outs: vec![v], op });
        v
    }
    fn ty(&self, v: V) -> &Ty {
        &self.types[v]
    }
    fn konst(&mut self, x: f64, dtype: DType) -> V {
        // Constants are emitted where first used; a constant first used
        // inside a loop body is local to it, so only top-level constants
        // are shared.
        if self.blocks.len() == 1
            && let Some(&v) = self.consts.get(&(x.to_bits(), dtype))
        {
            return v;
        }
        let v = self.emit(Op::Const(x), Ty::scalar(dtype));
        if self.blocks.len() == 1 {
            self.consts.insert((x.to_bits(), dtype), v);
        }
        v
    }

    /// A runtime value for `e`; literals take `hint`'s dtype if given.
    fn val(&mut self, e: &E, hint: Option<DType>, line: usize) -> Result<V, String> {
        match e {
            E::Val(v) => Ok(*v),
            E::Int(x) => {
                let dt = match hint {
                    Some(d) if d.is_float() => d,
                    _ => DType::I32,
                };
                Ok(self.konst(*x as f64, dt))
            }
            E::Float(x) => {
                let dt = match hint {
                    Some(d) if d.is_float() => d,
                    _ => DType::F32,
                };
                Ok(self.konst(*x, dt))
            }
            E::Arr(_) => err(line, "an array is not a value; index it to read a tile"),
            E::Shape(_) | E::DT(_) => err(line, "not a value"),
        }
    }

    fn cast(&mut self, v: V, dtype: DType) -> V {
        if self.ty(v).dtype == dtype {
            return v;
        }
        let shape = self.ty(v).shape.clone();
        self.emit(Op::Cast(v), Ty { shape, dtype })
    }

    /// `v` broadcast to `shape` (numpy rules: trailing dimensions align).
    fn broadcast(&mut self, v: V, shape: &[usize], line: usize) -> Result<V, String> {
        let t = self.ty(v).clone();
        if t.shape == shape {
            return Ok(v);
        }
        if t.is_scalar() {
            return Ok(self.emit(
                Op::Splat(v),
                Ty {
                    shape: shape.to_vec(),
                    dtype: t.dtype,
                },
            ));
        }
        let mut cur = v;
        let mut s = t.shape.clone();
        while s.len() < shape.len() {
            cur = self.emit(
                Op::ExpandDims(cur, 0),
                Ty {
                    shape: [vec![1], s.clone()].concat(),
                    dtype: t.dtype,
                },
            );
            s.insert(0, 1);
        }
        if s.len() != shape.len() {
            return err(line, format!("cannot broadcast {:?} to {shape:?}", t.shape));
        }
        for (a, b) in s.iter().zip(shape) {
            if a != b && *a != 1 {
                return err(line, format!("cannot broadcast {:?} to {shape:?}", t.shape));
            }
        }
        if s == shape {
            return Ok(cur);
        }
        Ok(self.emit(
            Op::Broadcast(cur),
            Ty {
                shape: shape.to_vec(),
                dtype: t.dtype,
            },
        ))
    }

    fn joint_shape(a: &[usize], b: &[usize], line: usize) -> Result<Vec<usize>, String> {
        let n = a.len().max(b.len());
        let mut out = vec![0; n];
        for i in 0..n {
            let x = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
            let y = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
            out[i] = if x == y || y == 1 {
                x
            } else if x == 1 {
                y
            } else {
                return err(line, format!("shapes {a:?} and {b:?} do not broadcast"));
            };
        }
        Ok(out)
    }

    fn promote(a: DType, b: DType) -> DType {
        use DType::*;
        match (a, b) {
            (F32, _) | (_, F32) => F32,
            (F16, _) | (_, F16) => F16,
            (I32, _) | (_, I32) => I32,
            _ => Bool,
        }
    }

    fn binary(&mut self, op: Bin, a: E, b: E, line: usize) -> Result<E, String> {
        // Fold compile-time integers.
        if let (E::Int(x), E::Int(y)) = (&a, &b) {
            let (x, y) = (*x, *y);
            let r = match op {
                Bin::Add => Some(x + y),
                Bin::Sub => Some(x - y),
                Bin::Mul => Some(x * y),
                Bin::FloorDiv if y != 0 => Some(x.div_euclid(y)),
                Bin::Mod if y != 0 => Some(x.rem_euclid(y)),
                Bin::Max => Some(x.max(y)),
                Bin::Min => Some(x.min(y)),
                _ => None,
            };
            if let Some(r) = r {
                return Ok(E::Int(r));
            }
        }
        let fa = |e: &E, s: &B| match e {
            E::Val(v) => Some(s.ty(*v).dtype),
            _ => None,
        };
        let (da, db) = (fa(&a, self), fa(&b, self));
        let lit_float = matches!(a, E::Float(_)) || matches!(b, E::Float(_));
        let mut dt = match (da, db) {
            (Some(x), Some(y)) => Self::promote(x, y),
            (Some(x), None) | (None, Some(x)) => {
                if lit_float && !x.is_float() {
                    DType::F32
                } else {
                    x
                }
            }
            (None, None) => {
                if lit_float {
                    DType::F32
                } else {
                    DType::I32
                }
            }
        };
        if op == Bin::Div && !dt.is_float() {
            dt = DType::F32;
        }
        if matches!(op, Bin::And | Bin::Or) && dt != DType::Bool {
            return err(line, "& and | need boolean operands (comparisons)");
        }
        let va = self.val(&a, Some(dt), line)?;
        let vb = self.val(&b, Some(dt), line)?;
        let va = self.cast(va, dt);
        let vb = self.cast(vb, dt);
        let shape = Self::joint_shape(&self.ty(va).shape.clone(), &self.ty(vb).shape.clone(), line)?;
        let va = self.broadcast(va, &shape, line)?;
        let vb = self.broadcast(vb, &shape, line)?;
        let out_dt = if op.is_compare() { DType::Bool } else { dt };
        if matches!(op, Bin::FloorDiv | Bin::Mod) && dt.is_float() {
            return err(line, "// and % are for integers");
        }
        Ok(E::Val(self.emit(Op::Binary(op, va, vb), Ty { shape, dtype: out_dt })))
    }

    fn const_int(&self, e: &E, line: usize) -> Result<i64, String> {
        match e {
            E::Int(x) => Ok(*x),
            _ => err(
                line,
                "expected a compile-time integer (array dimensions, meta constants and literals)",
            ),
        }
    }

    fn shape_of(&self, e: &E, line: usize) -> Result<Vec<usize>, String> {
        match e {
            E::Shape(s) => Ok(s.clone()),
            E::Int(x) => Ok(vec![*x as usize]),
            _ => err(line, "expected a shape such as [BM, BN]"),
        }
    }

    fn check_tile_shape(shape: &[usize], line: usize) -> Result<(), String> {
        if shape.len() > 2 {
            return err(line, "tiles have at most two dimensions");
        }
        for &d in shape {
            if d == 0 || !d.is_power_of_two() {
                return err(line, format!("tile dimensions must be powers of two, got {shape:?}"));
            }
        }
        Ok(())
    }

    fn index_arr(&mut self, arr: usize, idx: &[Index], line: usize) -> Result<(Vec<Ix>, Vec<usize>), String> {
        let PKind::Array { dims, .. } = self.params[arr].kind.clone() else {
            return err(line, "only arrays can be indexed this way");
        };
        if idx.len() != dims.len() {
            return err(
                line,
                format!(
                    "{} has {} dimensions, indexed with {}",
                    self.params[arr].name,
                    dims.len(),
                    idx.len()
                ),
            );
        }
        let mut out = Vec::new();
        let mut shape = Vec::new();
        for (d, ix) in idx.iter().enumerate() {
            match ix {
                Index::Point(e) => {
                    let e = self.expr(e)?;
                    let v = self.val(&e, Some(DType::I32), line)?;
                    if !self.ty(v).is_scalar() || self.ty(v).dtype != DType::I32 {
                        return err(line, "an array index must be an integer scalar");
                    }
                    out.push(Ix::Point(v));
                }
                Index::Slice(s, n) => {
                    let s = self.expr(s)?;
                    let n = self.expr(n)?;
                    let n = self.const_int(&n, line)? as usize;
                    let v = self.val(&s, Some(DType::I32), line)?;
                    if !self.ty(v).is_scalar() || self.ty(v).dtype != DType::I32 {
                        return err(line, "a slice start must be an integer scalar");
                    }
                    out.push(Ix::Slice(v, n));
                    shape.push(n);
                }
                Index::Full => {
                    let z = self.konst(0.0, DType::I32);
                    out.push(Ix::Slice(z, dims[d]));
                    shape.push(dims[d]);
                }
                Index::NewAxis => return err(line, "None indexes tiles, not arrays"),
            }
        }
        Self::check_tile_shape(&shape, line)?;
        Ok((out, shape))
    }

    fn load(&mut self, arr: usize, idx: &[Index], other: f64, line: usize) -> Result<E, String> {
        let (ix, shape) = self.index_arr(arr, idx, line)?;
        let PKind::Array { dtype, .. } = self.params[arr].kind else {
            unreachable!()
        };
        Ok(E::Val(self.emit(Op::Load { arr, idx: ix, other }, Ty { shape, dtype })))
    }

    fn unary(&mut self, op: Un, e: E, line: usize) -> Result<E, String> {
        let mut v = self.val(&e, None, line)?;
        if matches!(op, Un::Exp | Un::Log | Un::Sqrt | Un::Rsqrt) && !self.ty(v).dtype.is_float() {
            v = self.cast(v, DType::F32);
        }
        if op == Un::Not && self.ty(v).dtype != DType::Bool {
            return err(line, "'not' needs a boolean");
        }
        let t = self.ty(v).clone();
        Ok(E::Val(self.emit(Op::Unary(op, v), t)))
    }

    fn call(&mut self, name: &str, args: &[Expr], kw: &[(String, Expr)], line: usize) -> Result<E, String> {
        let kwarg = |k: &str| kw.iter().find(|(n, _)| n == k).map(|x| &x.1);
        let nargs = |n: usize| -> Result<(), String> {
            if args.len() == n {
                Ok(())
            } else {
                err(line, format!("{name} takes {n} arguments"))
            }
        };
        match name {
            "program_id" => {
                nargs(1)?;
                let a = self.expr(&args[0])?;
                let a = self.const_int(&a, line)?;
                if !(0..3).contains(&a) {
                    return err(line, "program_id axis is 0, 1 or 2");
                }
                Ok(E::Val(self.emit(Op::ProgramId(a as usize), Ty::scalar(DType::I32))))
            }
            "cdiv" => {
                nargs(2)?;
                let a = self.expr(&args[0])?;
                let b = self.expr(&args[1])?;
                if let (E::Int(x), E::Int(y)) = (&a, &b) {
                    return Ok(E::Int((x + y - 1).div_euclid(*y)));
                }
                let bm1 = self.binary(Bin::Sub, b.clone(), E::Int(1), line)?;
                let s = self.binary(Bin::Add, a, bm1, line)?;
                self.binary(Bin::FloorDiv, s, b, line)
            }
            "arange" => {
                nargs(1)?;
                let n = self.expr(&args[0])?;
                let n = self.const_int(&n, line)? as usize;
                Self::check_tile_shape(&[n], line)?;
                Ok(E::Val(self.emit(
                    Op::Arange,
                    Ty {
                        shape: vec![n],
                        dtype: DType::I32,
                    },
                )))
            }
            "zeros" | "full" => {
                let (shape, value, dt) = if name == "zeros" {
                    nargs(2)?;
                    (self.expr(&args[0])?, E::Float(0.0), self.expr(&args[1])?)
                } else {
                    nargs(3)?;
                    (self.expr(&args[0])?, self.expr(&args[1])?, self.expr(&args[2])?)
                };
                let shape = self.shape_of(&shape, line)?;
                Self::check_tile_shape(&shape, line)?;
                let E::DT(dt) = dt else {
                    return err(line, "expected a dtype such as f32");
                };
                let v = self.val(&value, Some(dt), line)?;
                let v = self.cast(v, dt);
                Ok(E::Val(self.emit(Op::Splat(v), Ty { shape, dtype: dt })))
            }
            "load" => {
                if args.len() != 1 {
                    return err(line, "load takes one block, as in load(A[i : +BM, 0 : +BK], other=0.0)");
                }
                let Expr::Index(a, idx, _) = &args[0] else {
                    return err(line, "load takes an indexed array");
                };
                let arr = self.expr(a)?;
                let E::Arr(arr) = arr else {
                    return err(line, "load takes an indexed array");
                };
                let other = match kwarg("other") {
                    Some(e) => match self.expr(e)? {
                        E::Int(x) => x as f64,
                        E::Float(x) => x,
                        _ => return err(line, "other must be a constant"),
                    },
                    None => 0.0,
                };
                self.load(arr, idx, other, line)
            }
            "exp" | "log" | "sqrt" | "rsqrt" | "abs" => {
                nargs(1)?;
                let e = self.expr(&args[0])?;
                let op = match name {
                    "exp" => Un::Exp,
                    "log" => Un::Log,
                    "sqrt" => Un::Sqrt,
                    "rsqrt" => Un::Rsqrt,
                    _ => Un::Abs,
                };
                self.unary(op, e, line)
            }
            "maximum" | "minimum" => {
                nargs(2)?;
                let a = self.expr(&args[0])?;
                let b = self.expr(&args[1])?;
                self.binary(if name == "maximum" { Bin::Max } else { Bin::Min }, a, b, line)
            }
            "where" => {
                nargs(3)?;
                let c = self.expr(&args[0])?;
                let a = self.expr(&args[1])?;
                let b = self.expr(&args[2])?;
                let c = self.val(&c, None, line)?;
                if self.ty(c).dtype != DType::Bool {
                    return err(line, "where's condition must be boolean");
                }
                let da = match &a {
                    E::Val(v) => Some(self.ty(*v).dtype),
                    _ => None,
                };
                let db = match &b {
                    E::Val(v) => Some(self.ty(*v).dtype),
                    _ => None,
                };
                let dt = match (da, db) {
                    (Some(x), Some(y)) => Self::promote(x, y),
                    (Some(x), None) | (None, Some(x)) => x,
                    (None, None) => DType::F32,
                };
                let va = self.val(&a, Some(dt), line)?;
                let vb = self.val(&b, Some(dt), line)?;
                let va = self.cast(va, dt);
                let vb = self.cast(vb, dt);
                let s1 = Self::joint_shape(&self.ty(va).shape.clone(), &self.ty(vb).shape.clone(), line)?;
                let shape = Self::joint_shape(&s1, &self.ty(c).shape.clone(), line)?;
                let c = self.broadcast(c, &shape, line)?;
                let va = self.broadcast(va, &shape, line)?;
                let vb = self.broadcast(vb, &shape, line)?;
                Ok(E::Val(self.emit(Op::Where(c, va, vb), Ty { shape, dtype: dt })))
            }
            "sum" | "max" | "min" => {
                let x = self.expr(&args[0])?;
                let x = self.val(&x, None, line)?;
                let axis = match (args.get(1), kwarg("axis")) {
                    (Some(e), _) | (None, Some(e)) => {
                        let a = self.expr(e)?;
                        self.const_int(&a, line)?
                    }
                    (None, None) => return err(line, format!("{name} needs an axis, as in {name}(x, axis=1)")),
                };
                let t = self.ty(x).clone();
                let rank = t.shape.len() as i64;
                if rank == 0 {
                    return err(line, "cannot reduce a scalar");
                }
                let axis = if axis < 0 { axis + rank } else { axis };
                if !(0..rank).contains(&axis) {
                    return err(line, format!("axis {axis} out of range for shape {:?}", t.shape));
                }
                let mut shape = t.shape.clone();
                shape.remove(axis as usize);
                let op = match name {
                    "sum" => Red::Sum,
                    "max" => Red::Max,
                    _ => Red::Min,
                };
                let x = if op == Red::Sum && t.dtype == DType::F16 {
                    self.cast(x, DType::F32)
                } else {
                    x
                };
                let dtype = self.ty(x).dtype;
                Ok(E::Val(self.emit(Op::Reduce(op, x, axis as usize), Ty { shape, dtype })))
            }
            "trans" => {
                nargs(1)?;
                let x = self.expr(&args[0])?;
                let x = self.val(&x, None, line)?;
                let t = self.ty(x).clone();
                if t.shape.len() != 2 {
                    return err(line, "trans needs a 2-D tile");
                }
                Ok(E::Val(self.emit(
                    Op::Trans(x),
                    Ty {
                        shape: vec![t.shape[1], t.shape[0]],
                        dtype: t.dtype,
                    },
                )))
            }
            "dot" => {
                if args.len() != 2 && args.len() != 3 {
                    return err(line, "dot takes (a, b) or (a, b, acc)");
                }
                let a = self.expr(&args[0])?;
                let a = self.val(&a, None, line)?;
                let b = self.expr(&args[1])?;
                let b = self.val(&b, None, line)?;
                let (ta, tb) = (self.ty(a).clone(), self.ty(b).clone());
                if ta.shape.len() != 2 || tb.shape.len() != 2 || ta.shape[1] != tb.shape[0] {
                    return err(line, format!("dot of {:?} and {:?}", ta.shape, tb.shape));
                }
                if ta.dtype != tb.dtype || !ta.dtype.is_float() {
                    return err(line, "dot needs two f16 or two f32 tiles");
                }
                if ta.shape[0] < 16 || ta.shape[1] < 16 || tb.shape[1] < 16 {
                    return err(line, "dot tiles must be at least 16 x 16");
                }
                let shape = vec![ta.shape[0], tb.shape[1]];
                let acc = if args.len() == 3 {
                    let c = self.expr(&args[2])?;
                    let c = self.val(&c, None, line)?;
                    if self.ty(c).shape != shape || self.ty(c).dtype != DType::F32 {
                        return err(line, format!("dot's accumulator must be f32 {shape:?}"));
                    }
                    c
                } else {
                    let z = self.konst(0.0, DType::F32);
                    self.emit(
                        Op::Splat(z),
                        Ty {
                            shape: shape.clone(),
                            dtype: DType::F32,
                        },
                    )
                };
                Ok(E::Val(self.emit(
                    Op::Dot(a, b, acc),
                    Ty {
                        shape,
                        dtype: DType::F32,
                    },
                )))
            }
            "f16" | "f32" | "i32" | "cast" => {
                let (x, dt) = if name == "cast" {
                    nargs(2)?;
                    let d = self.expr(&args[1])?;
                    let E::DT(d) = d else {
                        return err(line, "cast(x, dtype)");
                    };
                    (self.expr(&args[0])?, d)
                } else {
                    nargs(1)?;
                    (self.expr(&args[0])?, DType::parse(name).unwrap())
                };
                let v = self.val(&x, Some(dt), line)?;
                Ok(E::Val(self.cast(v, dt)))
            }
            _ => err(line, format!("unknown function {name}")),
        }
    }

    fn expr(&mut self, e: &Expr) -> Result<E, String> {
        match e {
            Expr::Int(x) => Ok(E::Int(*x)),
            Expr::Float(x) => Ok(E::Float(*x)),
            Expr::Name(n, line) => {
                if let Some(d) = DType::parse(n) {
                    return Ok(E::DT(d));
                }
                self.env.get(n).cloned().ok_or(format!("line {line}: unknown name {n}"))
            }
            Expr::List(v, line) => {
                let mut s = Vec::new();
                for x in v {
                    let x = self.expr(x)?;
                    s.push(self.const_int(&x, *line)? as usize);
                }
                Ok(E::Shape(s))
            }
            Expr::Neg(x, line) => {
                let x = self.expr(x)?;
                match x {
                    E::Int(v) => Ok(E::Int(-v)),
                    E::Float(v) => Ok(E::Float(-v)),
                    x => self.unary(Un::Neg, x, *line),
                }
            }
            Expr::Bin(op, a, b, line) => {
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                let op = match op {
                    AB::Add => Bin::Add,
                    AB::Sub => Bin::Sub,
                    AB::Mul => Bin::Mul,
                    AB::Div => Bin::Div,
                    AB::FloorDiv => Bin::FloorDiv,
                    AB::Mod => Bin::Mod,
                    AB::Lt => Bin::Lt,
                    AB::Le => Bin::Le,
                    AB::Gt => Bin::Gt,
                    AB::Ge => Bin::Ge,
                    AB::Eq => Bin::Eq,
                    AB::Ne => Bin::Ne,
                    AB::And => Bin::And,
                    AB::Or => Bin::Or,
                };
                self.binary(op, a, b, *line)
            }
            Expr::Call(n, args, kw, line) => self.call(n, args, kw, *line),
            Expr::Index(base, idx, line) => {
                let b = self.expr(base)?;
                match b {
                    E::Arr(a) => self.load(a, idx, 0.0, *line),
                    E::Val(v) => {
                        // tile[:, None] and tile[None, :]
                        let t = self.ty(v).clone();
                        let mut cur = v;
                        let mut shape = t.shape.clone();
                        let mut d = 0;
                        for ix in idx {
                            match ix {
                                Index::Full => d += 1,
                                Index::NewAxis => {
                                    shape.insert(d, 1);
                                    cur = self.emit(
                                        Op::ExpandDims(cur, d),
                                        Ty {
                                            shape: shape.clone(),
                                            dtype: t.dtype,
                                        },
                                    );
                                    d += 1;
                                }
                                _ => return err(*line, "tiles are indexed only with : and None"),
                            }
                        }
                        if d != shape.len() || shape.len() > 2 {
                            return err(*line, "bad tile indexing");
                        }
                        Ok(E::Val(cur))
                    }
                    _ => err(*line, "cannot index this"),
                }
            }
        }
    }

    fn assigned(stmts: &[Stmt], out: &mut Vec<String>) {
        for s in stmts {
            match s {
                Stmt::Assign(n, _, _) => {
                    if !out.contains(n) {
                        out.push(n.clone());
                    }
                }
                Stmt::For { body, .. } => Self::assigned(body, out),
                _ => {}
            }
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for s in stmts {
            match s {
                Stmt::Meta(_, line) => {
                    if self.blocks.len() > 1 {
                        return err(*line, "meta belongs at the top of the kernel");
                    }
                }
                Stmt::Grid(es, line) => {
                    let mut g = [1usize; 3];
                    if es.is_empty() || es.len() > 3 {
                        return err(*line, "grid has 1 to 3 dimensions");
                    }
                    for (i, e) in es.iter().enumerate() {
                        let x = self.expr(e)?;
                        let x = self.const_int(&x, *line)?;
                        if x < 1 {
                            return err(*line, "grid dimensions must be positive");
                        }
                        g[i] = x as usize;
                    }
                    self.grid = Some(g);
                }
                Stmt::Assign(n, e, _) => {
                    let v = self.expr(e)?;
                    self.env.insert(n.clone(), v);
                }
                Stmt::Store(n, idx, e, line) => {
                    let Some(E::Arr(arr)) = self.env.get(n).cloned() else {
                        return err(*line, format!("{n} is not an array"));
                    };
                    let (ix, shape) = self.index_arr(arr, idx, *line)?;
                    let PKind::Array { dtype, .. } = self.params[arr].kind else {
                        unreachable!()
                    };
                    let v = self.expr(e)?;
                    let v = self.val(&v, Some(dtype), *line)?;
                    let v = self.cast(v, dtype);
                    let v = self.broadcast(v, &shape, *line)?;
                    self.blocks.last_mut().unwrap().push(Inst {
                        outs: vec![],
                        op: Op::Store { arr, idx: ix, value: v },
                    });
                }
                Stmt::For {
                    var,
                    start,
                    end,
                    step,
                    body,
                    line,
                } => {
                    let s = self.expr(start)?;
                    let e = self.expr(end)?;
                    let st = self.expr(step)?;
                    if let E::Int(x) = st
                        && x <= 0
                    {
                        return err(*line, "range step must be positive");
                    }
                    let (vs, ve, vst) = (
                        self.val(&s, Some(DType::I32), *line)?,
                        self.val(&e, Some(DType::I32), *line)?,
                        self.val(&st, Some(DType::I32), *line)?,
                    );
                    let mut names = Vec::new();
                    Self::assigned(body, &mut names);
                    let carried: Vec<String> = names
                        .into_iter()
                        .filter(|n| self.env.contains_key(n) && n != var)
                        .collect();
                    let mut init = Vec::new();
                    let mut args = Vec::new();
                    let saved = self.env.clone();
                    for n in &carried {
                        let cur = self.env[n].clone();
                        let v = self.val(&cur, None, *line)?;
                        init.push(v);
                        let a = self.new_val(self.ty(v).clone());
                        args.push(a);
                        self.env.insert(n.clone(), E::Val(a));
                    }
                    let iv = self.new_val(Ty::scalar(DType::I32));
                    self.env.insert(var.clone(), E::Val(iv));
                    self.blocks.push(Vec::new());
                    self.stmts(body)?;
                    let mut yields = Vec::new();
                    for (k, n) in carried.iter().enumerate() {
                        let cur = self.env[n].clone();
                        let want = self.ty(args[k]).clone();
                        let v = self.val(&cur, Some(want.dtype), *line)?;
                        let v = if self.ty(v).dtype != want.dtype && self.ty(v).is_scalar() == want.is_scalar() {
                            self.cast(v, want.dtype)
                        } else {
                            v
                        };
                        let v = if want.is_scalar() {
                            v
                        } else {
                            self.broadcast(v, &want.shape, *line)?
                        };
                        if *self.ty(v) != want {
                            return err(
                                *line,
                                format!("{n} changes type in the loop ({:?} to {:?})", want, self.ty(v)),
                            );
                        }
                        yields.push(v);
                    }
                    let body = self.blocks.pop().unwrap();
                    self.env = saved;
                    let outs: Vec<V> = args.iter().map(|&a| self.new_val(self.ty(a).clone())).collect();
                    for (n, &o) in carried.iter().zip(&outs) {
                        self.env.insert(n.clone(), E::Val(o));
                    }
                    self.blocks.last_mut().unwrap().push(Inst {
                        outs,
                        op: Op::For {
                            iv,
                            start: vs,
                            end: ve,
                            step: vst,
                            args,
                            init,
                            body,
                            yields,
                        },
                    });
                }
            }
        }
        Ok(())
    }
}

/// The shapes and meta overrides a kernel is compiled for.
#[derive(Clone, Debug, Default)]
pub struct Spec {
    /// Every array parameter's shape, in parameter order.
    pub shapes: Vec<Vec<usize>>,
    pub meta: Vec<(String, i64)>,
}

/// Specializes `def` for `spec`.
pub fn build(def: &KernelDef, spec: &Spec) -> Result<Kernel, String> {
    let mut dims: HashMap<String, i64> = HashMap::new();
    let mut params = Vec::new();
    let mut env = HashMap::new();
    let mut si = 0;
    for (pi, p) in def.params.iter().enumerate() {
        match &p.ty {
            ParamTy::Array { dtype, dims: ds } => {
                let shape = spec
                    .shapes
                    .get(si)
                    .ok_or(format!("{}: no shape given for array {}", def.name, p.name))?;
                si += 1;
                if shape.len() != ds.len() {
                    return Err(format!(
                        "{}: {} has {} dimensions, shape {:?} given",
                        def.name,
                        p.name,
                        ds.len(),
                        shape
                    ));
                }
                for (d, &n) in ds.iter().zip(shape) {
                    match d {
                        DimSpec::Int(x) if *x as usize != n => {
                            return Err(format!("{}: {} dimension must be {x}, got {n}", def.name, p.name));
                        }
                        DimSpec::Name(name) => {
                            if let Some(&old) = dims.get(name)
                                && old as usize != n
                            {
                                return Err(format!("{}: dimension {name} is {old} and {n}", def.name));
                            }
                            dims.insert(name.clone(), n as i64);
                        }
                        _ => {}
                    }
                }
                params.push(IrParam {
                    name: p.name.clone(),
                    kind: PKind::Array {
                        dtype: *dtype,
                        dims: shape.clone(),
                    },
                });
                env.insert(p.name.clone(), E::Arr(pi));
            }
            ParamTy::Scalar(dt) => {
                params.push(IrParam {
                    name: p.name.clone(),
                    kind: PKind::Scalar(*dt),
                });
            }
        }
    }
    for (n, v) in &dims {
        env.insert(n.clone(), E::Int(*v));
    }
    // Meta constants: declared defaults, overridden by the spec.
    let mut meta = Vec::new();
    for s in &def.body {
        if let Stmt::Meta(v, _) = s {
            for (n, x) in v {
                let x = spec.meta.iter().find(|(m, _)| m == n).map(|m| m.1).unwrap_or(*x);
                meta.push((n.clone(), x));
                env.insert(n.clone(), E::Int(x));
            }
        }
    }
    for (n, _) in &spec.meta {
        if !meta.iter().any(|(m, _)| m == n) {
            return Err(format!("{}: no meta constant {n}", def.name));
        }
    }
    let mut b = B {
        def,
        params,
        types: Vec::new(),
        blocks: vec![Vec::new()],
        env,
        meta,
        grid: None,
        consts: HashMap::new(),
    };
    // Scalar parameters become values up front.
    for (pi, p) in def.params.iter().enumerate() {
        if let ParamTy::Scalar(dt) = p.ty {
            let v = b.emit(Op::ScalarArg(pi), Ty::scalar(dt));
            b.env.insert(p.name.clone(), E::Val(v));
        }
    }
    b.stmts(&def.body)?;
    let grid = b.grid.ok_or(format!("{}: no grid(...) statement", b.def.name))?;
    Ok(Kernel {
        name: def.name.clone(),
        params: b.params,
        meta: b.meta,
        grid,
        types: b.types,
        body: b.blocks.pop().unwrap(),
    })
}

/// Parses `src` and specializes its kernel named `name`.
pub fn compile(src: &str, name: &str, spec: &Spec) -> Result<Kernel, String> {
    let defs = ast::parse(src)?;
    let def = defs
        .iter()
        .find(|d| d.name == name)
        .ok_or(format!("no kernel {name}"))?;
    build(def, spec)
}

/// Prints the IR, for debugging and the `ir` command.
pub fn dump(k: &Kernel) -> String {
    fn ty(t: &Ty) -> String {
        if t.shape.is_empty() {
            t.dtype.name().to_string()
        } else {
            format!("{}{:?}", t.dtype.name(), t.shape)
        }
    }
    fn block(k: &Kernel, b: &[Inst], ind: usize, out: &mut String) {
        for i in b {
            let pad = " ".repeat(ind);
            let lhs = i
                .outs
                .iter()
                .map(|v| format!("%{v}: {}", ty(&k.types[*v])))
                .collect::<Vec<_>>()
                .join(", ");
            match &i.op {
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
                    out.push_str(&format!(
                        "{pad}{lhs} = for %{iv} in range(%{start}, %{end}, %{step}) carrying {}:\n",
                        args.iter()
                            .zip(init)
                            .map(|(a, x)| format!("%{a} = %{x}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                    block(k, body, ind + 4, out);
                    out.push_str(&format!(
                        "{pad}    yield {}\n",
                        yields.iter().map(|y| format!("%{y}")).collect::<Vec<_>>().join(", ")
                    ));
                }
                op => {
                    if lhs.is_empty() {
                        out.push_str(&format!("{pad}{op:?}\n"));
                    } else {
                        out.push_str(&format!("{pad}{lhs} = {op:?}\n"));
                    }
                }
            }
        }
    }
    let mut s = format!("kernel {} grid {:?} meta {:?}\n", k.name, k.grid, k.meta);
    block(k, &k.body, 4, &mut s);
    s
}
