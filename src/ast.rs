//! Syntax of the tile language, and its parser.
//!
//! ```text
//! kernel matmul(A: f16[M, K], B: f16[K, N], C: f32[M, N]):
//!     meta BM = 64, BN = 64, BK = 32
//!     grid(cdiv(M, BM), cdiv(N, BN))
//!     i = program_id(0)
//!     j = program_id(1)
//!     acc = zeros([BM, BN], f32)
//!     for k in range(0, K, BK):
//!         acc = dot(A[i * BM : +BM, k : +BK], B[k : +BK, j * BN : +BN], acc)
//!     C[i * BM : +BM, j * BN : +BN] = acc
//! ```
//!
//! A kernel runs once per point of its grid (a *program*), and works on
//! *tiles*: small rectangular blocks of values with sizes fixed at compile
//! time. `A[s : +n, ...]` reads the block of `n` elements starting at `s`
//! (elements past the end of the array read as 0, or as `other` given to
//! `load`); assigning to such a block writes it (elements past the end are
//! skipped). Array dimensions (`M`, `K`) and `meta` constants are known
//! when the kernel is compiled.

use crate::lexer::{Tok, Token, lex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    F32,
    I32,
    Bool,
}

impl DType {
    pub fn parse(s: &str) -> Option<DType> {
        Some(match s {
            "f16" => DType::F16,
            "f32" => DType::F32,
            "i32" => DType::I32,
            "bool" => DType::Bool,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::I32 => "i32",
            DType::Bool => "bool",
        }
    }
    pub fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::F32)
    }
    pub fn bytes(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::Bool => 1,
            _ => 4,
        }
    }
}

#[derive(Clone, Debug)]
pub enum DimSpec {
    Name(String),
    Int(i64),
}

#[derive(Clone, Debug)]
pub enum ParamTy {
    Array { dtype: DType, dims: Vec<DimSpec> },
    Scalar(DType),
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: ParamTy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Clone, Debug)]
pub enum Index {
    Point(Expr),
    Slice(Expr, Expr),
    Full,
    NewAxis,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Name(String, usize),
    Int(i64),
    Float(f64),
    Bin(BinOp, Box<Expr>, Box<Expr>, usize),
    Neg(Box<Expr>, usize),
    Call(String, Vec<Expr>, Vec<(String, Expr)>, usize),
    Index(Box<Expr>, Vec<Index>, usize),
    List(Vec<Expr>, usize),
}

impl Expr {
    pub fn line(&self) -> usize {
        match self {
            Expr::Name(_, l)
            | Expr::Bin(_, _, _, l)
            | Expr::Neg(_, l)
            | Expr::Call(_, _, _, l)
            | Expr::Index(_, _, l)
            | Expr::List(_, l) => *l,
            Expr::Int(_) | Expr::Float(_) => 0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Meta(Vec<(String, i64)>, usize),
    Grid(Vec<Expr>, usize),
    Assign(String, Expr, usize),
    Store(String, Vec<Index>, Expr, usize),
    For {
        var: String,
        start: Expr,
        end: Expr,
        step: Expr,
        body: Vec<Stmt>,
        line: usize,
    },
}

#[derive(Clone, Debug)]
pub struct KernelDef {
    pub name: String,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
}

struct P {
    t: Vec<Token>,
    i: usize,
}

impl P {
    fn peek(&self) -> &Tok {
        &self.t[self.i].tok
    }
    fn line(&self) -> usize {
        self.t[self.i].line
    }
    fn next(&mut self) -> Tok {
        let t = self.t[self.i].tok.clone();
        if self.i + 1 < self.t.len() {
            self.i += 1;
        }
        t
    }
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!(
            "line {}: expected {what}, found {:?}",
            self.line(),
            self.peek()
        ))
    }
    fn is(&self, s: &str) -> bool {
        matches!(self.peek(), Tok::Sym(x) if *x == s)
    }
    fn is_name(&self, s: &str) -> bool {
        matches!(self.peek(), Tok::Name(x) if x == s)
    }
    fn eat(&mut self, s: &str) -> bool {
        if self.is(s) {
            self.next();
            true
        } else {
            false
        }
    }
    fn expect(&mut self, s: &str) -> Result<(), String> {
        if self.eat(s) {
            Ok(())
        } else {
            self.err(&format!("'{s}'"))
        }
    }
    fn name(&mut self) -> Result<String, String> {
        match self.peek().clone() {
            Tok::Name(n) => {
                self.next();
                Ok(n)
            }
            _ => self.err("a name"),
        }
    }
    fn newline(&mut self) -> Result<(), String> {
        match self.peek() {
            Tok::Newline => {
                self.next();
                Ok(())
            }
            _ => self.err("end of line"),
        }
    }

    fn kernel(&mut self) -> Result<KernelDef, String> {
        if !self.is_name("kernel") {
            return self.err("'kernel'");
        }
        self.next();
        let name = self.name()?;
        self.expect("(")?;
        let mut params = Vec::new();
        while !self.is(")") {
            let pname = self.name()?;
            self.expect(":")?;
            let tname = self.name()?;
            let dtype = DType::parse(&tname).ok_or(format!("line {}: unknown type {tname}", self.line()))?;
            let ty = if self.eat("[") {
                let mut dims = Vec::new();
                while !self.is("]") {
                    dims.push(match self.next() {
                        Tok::Name(n) => DimSpec::Name(n),
                        Tok::Int(v) => DimSpec::Int(v),
                        _ => return self.err("a dimension"),
                    });
                    if !self.eat(",") {
                        break;
                    }
                }
                self.expect("]")?;
                ParamTy::Array { dtype, dims }
            } else {
                ParamTy::Scalar(dtype)
            };
            params.push(Param { name: pname, ty });
            if !self.eat(",") {
                break;
            }
        }
        self.expect(")")?;
        self.expect(":")?;
        self.newline()?;
        let body = self.block()?;
        Ok(KernelDef { name, params, body })
    }

    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        if *self.peek() != Tok::Indent {
            return self.err("an indented block");
        }
        self.next();
        let mut out = Vec::new();
        while *self.peek() != Tok::Dedent && *self.peek() != Tok::Eof {
            out.push(self.stmt()?);
        }
        if *self.peek() == Tok::Dedent {
            self.next();
        }
        Ok(out)
    }

    fn stmt(&mut self) -> Result<Stmt, String> {
        let line = self.line();
        if self.is_name("meta") {
            self.next();
            let mut v = Vec::new();
            loop {
                let n = self.name()?;
                self.expect("=")?;
                let neg = self.eat("-");
                let x = match self.next() {
                    Tok::Int(x) => x,
                    _ => return self.err("an integer"),
                };
                v.push((n, if neg { -x } else { x }));
                if !self.eat(",") {
                    break;
                }
            }
            self.newline()?;
            return Ok(Stmt::Meta(v, line));
        }
        if self.is_name("grid") {
            self.next();
            self.expect("(")?;
            let mut v = Vec::new();
            while !self.is(")") {
                v.push(self.expr()?);
                if !self.eat(",") {
                    break;
                }
            }
            self.expect(")")?;
            self.newline()?;
            return Ok(Stmt::Grid(v, line));
        }
        if self.is_name("for") {
            self.next();
            let var = self.name()?;
            if !self.is_name("in") {
                return self.err("'in'");
            }
            self.next();
            if !self.is_name("range") {
                return self.err("'range'");
            }
            self.next();
            self.expect("(")?;
            let a = self.expr()?;
            let (start, end, step) = if self.eat(",") {
                let b = self.expr()?;
                let c = if self.eat(",") { self.expr()? } else { Expr::Int(1) };
                (a, b, c)
            } else {
                (Expr::Int(0), a, Expr::Int(1))
            };
            self.expect(")")?;
            self.expect(":")?;
            self.newline()?;
            let body = self.block()?;
            return Ok(Stmt::For {
                var,
                start,
                end,
                step,
                body,
                line,
            });
        }
        let name = self.name()?;
        if self.eat("[") {
            let idx = self.indices()?;
            self.expect("=")?;
            let e = self.expr()?;
            self.newline()?;
            return Ok(Stmt::Store(name, idx, e, line));
        }
        for (s, op) in [("+=", BinOp::Add), ("-=", BinOp::Sub), ("*=", BinOp::Mul)] {
            if self.eat(s) {
                let e = self.expr()?;
                self.newline()?;
                let cur = Expr::Name(name.clone(), line);
                return Ok(Stmt::Assign(
                    name,
                    Expr::Bin(op, Box::new(cur), Box::new(e), line),
                    line,
                ));
            }
        }
        self.expect("=")?;
        let e = self.expr()?;
        self.newline()?;
        Ok(Stmt::Assign(name, e, line))
    }

    /// After '[': indices up to and including ']'.
    fn indices(&mut self) -> Result<Vec<Index>, String> {
        let mut v = Vec::new();
        while !self.is("]") {
            if self.is(":") {
                self.next();
                v.push(Index::Full);
            } else if self.is_name("None") {
                self.next();
                v.push(Index::NewAxis);
            } else {
                let a = self.expr()?;
                if self.eat(":") {
                    self.expect("+")?;
                    let n = self.expr()?;
                    v.push(Index::Slice(a, n));
                } else {
                    v.push(Index::Point(a));
                }
            }
            if !self.eat(",") {
                break;
            }
        }
        self.expect("]")?;
        Ok(v)
    }

    fn expr(&mut self) -> Result<Expr, String> {
        self.binary(0)
    }

    fn binary(&mut self, level: usize) -> Result<Expr, String> {
        const LEVELS: [&[(&str, BinOp)]; 5] = [
            &[("|", BinOp::Or)],
            &[("&", BinOp::And)],
            &[
                ("<=", BinOp::Le),
                (">=", BinOp::Ge),
                ("==", BinOp::Eq),
                ("!=", BinOp::Ne),
                ("<", BinOp::Lt),
                (">", BinOp::Gt),
            ],
            &[("+", BinOp::Add), ("-", BinOp::Sub)],
            &[
                ("*", BinOp::Mul),
                ("//", BinOp::FloorDiv),
                ("/", BinOp::Div),
                ("%", BinOp::Mod),
            ],
        ];
        if level == LEVELS.len() {
            return self.unary();
        }
        let mut lhs = self.binary(level + 1)?;
        'outer: loop {
            for (s, op) in LEVELS[level] {
                if self.is(s) {
                    let line = self.line();
                    self.next();
                    let rhs = self.binary(level + 1)?;
                    lhs = Expr::Bin(*op, Box::new(lhs), Box::new(rhs), line);
                    continue 'outer;
                }
            }
            break;
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        let line = self.line();
        if self.eat("-") {
            let e = self.unary()?;
            return Ok(match e {
                Expr::Int(v) => Expr::Int(-v),
                Expr::Float(v) => Expr::Float(-v),
                e => Expr::Neg(Box::new(e), line),
            });
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.atom()?;
        loop {
            let line = self.line();
            if self.eat("[") {
                let idx = self.indices()?;
                e = Expr::Index(Box::new(e), idx, line);
            } else {
                break;
            }
        }
        Ok(e)
    }

    fn atom(&mut self) -> Result<Expr, String> {
        let line = self.line();
        match self.next() {
            Tok::Int(v) => Ok(Expr::Int(v)),
            Tok::Float(v) => Ok(Expr::Float(v)),
            Tok::Name(n) => {
                if n == "inf" {
                    return Ok(Expr::Float(f64::INFINITY));
                }
                if self.eat("(") {
                    let mut args = Vec::new();
                    let mut kw = Vec::new();
                    while !self.is(")") {
                        if let Tok::Name(k) = self.peek().clone()
                            && matches!(self.t[self.i + 1].tok, Tok::Sym("="))
                        {
                            self.next();
                            self.next();
                            kw.push((k, self.expr()?));
                        } else {
                            args.push(self.expr()?);
                        }
                        if !self.eat(",") {
                            break;
                        }
                    }
                    self.expect(")")?;
                    Ok(Expr::Call(n, args, kw, line))
                } else {
                    Ok(Expr::Name(n, line))
                }
            }
            Tok::Sym("(") => {
                let e = self.expr()?;
                self.expect(")")?;
                Ok(e)
            }
            Tok::Sym("[") => {
                let mut v = Vec::new();
                while !self.is("]") {
                    v.push(self.expr()?);
                    if !self.eat(",") {
                        break;
                    }
                }
                self.expect("]")?;
                Ok(Expr::List(v, line))
            }
            _ => {
                self.i -= 1;
                self.err("an expression")
            }
        }
    }
}

/// Parses every kernel in `src`.
pub fn parse(src: &str) -> Result<Vec<KernelDef>, String> {
    let mut p = P { t: lex(src)?, i: 0 };
    let mut out = Vec::new();
    while *p.peek() != Tok::Eof {
        if *p.peek() == Tok::Newline {
            p.next();
            continue;
        }
        out.push(p.kernel()?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_matmul() {
        let k = &parse(
            "kernel mm(A: f16[M, K], B: f16[K, N], C: f32[M, N]):\n    meta BM = 64, BN = 64, BK = 32\n    grid(cdiv(M, BM), cdiv(N, BN))\n    i = program_id(0)\n    acc = zeros([BM, BN], f32)\n    for k in range(0, K, BK):\n        acc = dot(A[i * BM : +BM, k : +BK], B[k : +BK, 0 : +BN], acc)\n    C[i * BM : +BM, 0 : +BN] = acc\n",
        )
        .unwrap()[0];
        assert_eq!(k.name, "mm");
        assert_eq!(k.params.len(), 3);
        assert_eq!(k.body.len(), 6);
        assert!(matches!(&k.body[4], Stmt::For { body, .. } if body.len() == 1));
        assert!(matches!(&k.body[5], Stmt::Store(n, idx, _, _) if n == "C" && idx.len() == 2));
    }

    #[test]
    fn precedence() {
        let k = &parse("kernel k(x: f32):\n    y = 1 + 2 * 3 < 4 & 5 > -6\n").unwrap()[0];
        let Stmt::Assign(_, e, _) = &k.body[0] else { panic!() };
        assert!(matches!(e, Expr::Bin(BinOp::And, ..)));
    }
}
