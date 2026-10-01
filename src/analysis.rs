//! Static facts about a kernel's integer scalars, shared by the backends.

use crate::ast::DType;
use crate::ir::{Bin, Inst, Kernel, Op};

/// The range of every integer scalar where it is known: constants,
/// program ids over the grid, loop indices over the values the loop runs,
/// and arithmetic on them.
pub fn ranges(k: &Kernel) -> Vec<Option<(i64, i64)>> {
    let mut rng = vec![None; k.types.len()];
    walk(k, &mut rng, &k.body);
    rng
}

/// `size` positions from integer scalar `v` provably lie in `0..dim`.
pub fn in_bounds(rng: &[Option<(i64, i64)>], v: usize, size: usize, dim: usize) -> bool {
    matches!(rng[v], Some((lo, hi)) if lo >= 0 && hi + size as i64 <= dim as i64)
}

fn walk(k: &Kernel, rng: &mut [Option<(i64, i64)>], b: &[Inst]) {
    for i in b {
        let out = i.outs.first().copied();
        match &i.op {
            Op::Const(x) if !k.ty(out.unwrap()).dtype.is_float() => {
                rng[out.unwrap()] = Some((*x as i64, *x as i64));
            }
            Op::ProgramId(a) => rng[out.unwrap()] = Some((0, k.grid[*a] as i64 - 1)),
            Op::Binary(op, a, b) if k.ty(out.unwrap()).is_scalar() && k.ty(out.unwrap()).dtype == DType::I32 => {
                if let (Some(x), Some(y)) = (rng[*a], rng[*b]) {
                    let fl = |p: i64, q: i64| p.div_euclid(q);
                    rng[out.unwrap()] = match op {
                        Bin::Add => Some((x.0 + y.0, x.1 + y.1)),
                        Bin::Sub => Some((x.0 - y.1, x.1 - y.0)),
                        Bin::Mul => {
                            let c = [x.0 * y.0, x.0 * y.1, x.1 * y.0, x.1 * y.1];
                            Some((*c.iter().min().unwrap(), *c.iter().max().unwrap()))
                        }
                        Bin::FloorDiv if y.0 > 0 => {
                            let c = [fl(x.0, y.0), fl(x.0, y.1), fl(x.1, y.0), fl(x.1, y.1)];
                            Some((*c.iter().min().unwrap(), *c.iter().max().unwrap()))
                        }
                        Bin::Mod if y.0 > 0 && x.0 >= 0 => Some((0, x.1.min(y.1 - 1))),
                        Bin::Max => Some((x.0.max(y.0), x.1.max(y.1))),
                        Bin::Min => Some((x.0.min(y.0), x.1.min(y.1))),
                        _ => None,
                    };
                }
            }
            Op::For {
                iv,
                start,
                end,
                step,
                body,
                args,
                ..
            } => {
                for a in args {
                    rng[*a] = None;
                }
                if let (Some(s), Some(e), Some(st)) = (rng[*start], rng[*end], rng[*step])
                    && st.0 == st.1
                    && st.0 > 0
                {
                    let last = if s.0 == s.1 {
                        s.0 + (e.1 - 1 - s.0).div_euclid(st.0) * st.0
                    } else {
                        e.1 - 1
                    };
                    rng[*iv] = Some((s.0, last.max(s.0)));
                }
                walk(k, rng, body);
            }
            _ => {}
        }
    }
}
