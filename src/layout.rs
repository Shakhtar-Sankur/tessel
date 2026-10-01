//! Linear layouts: how a tile's elements are spread over a program's
//! threads, as in Triton. A tile with a power-of-two number of elements
//! has positions `p = row * C + col`; a thread (warp `w`, lane `l`) holds
//! registers `k = 0 .. 2^reg.len()`, and register `k` of thread (w, l)
//! holds the element at
//!
//! ```text
//! p = XOR of reg[i] for bits i of k
//!   ^ XOR of lane[i] for bits i of l
//!   ^ XOR of warp[i] for bits i of w
//! ```
//!
//! A zero basis means replication: those threads (or registers) hold the
//! same elements. One representation covers coalesced blocked layouts,
//! tensor-core fragments, reduction results (slices) and transposes, and
//! makes the questions code generation asks (does this thread already hold
//! what that layout needs? which lanes take part in a reduction?) simple
//! bit arithmetic.

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub reg: Vec<u32>,
    pub lane: [u32; 5],
    pub warp: Vec<u32>,
}

fn log2(n: usize) -> u32 {
    debug_assert!(n.is_power_of_two());
    n.trailing_zeros()
}

/// Bits of `x` reduced against `basis` (an echelon basis kept sorted by
/// leading bit): zero exactly when `x` is in the span.
fn reduce(mut x: u32, basis: &[u32]) -> u32 {
    for &b in basis {
        let top = 31 - b.leading_zeros();
        if x >> top & 1 == 1 {
            x ^= b;
        }
    }
    x
}

fn insert(basis: &mut Vec<u32>, x: u32) -> bool {
    let r = reduce(x, basis);
    if r == 0 {
        return false;
    }
    basis.push(r);
    basis.sort_by_key(|b| std::cmp::Reverse(31 - b.leading_zeros()));
    true
}

impl Layout {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn nregs(&self) -> usize {
        1 << self.reg.len()
    }
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    /// Columns of the tile: the last dimension (1 for scalars).
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }
    pub fn reg_pos(&self, k: usize) -> u32 {
        (0..self.reg.len())
            .filter(|i| k >> i & 1 == 1)
            .fold(0, |a, i| a ^ self.reg[i])
    }
    pub fn thread_pos(&self, lane: usize, warp: usize) -> u32 {
        let mut p = 0;
        for i in 0..5 {
            if lane >> i & 1 == 1 {
                p ^= self.lane[i];
            }
        }
        for (i, b) in self.warp.iter().enumerate() {
            if warp >> i & 1 == 1 {
                p ^= b;
            }
        }
        p
    }
    /// (row, col) of position `p` (row 0 for a 1-D tile).
    pub fn coords(&self, p: u32) -> (usize, usize) {
        let c = self.cols();
        (p as usize / c, p as usize % c)
    }

    /// Every element is held by some thread.
    pub fn covers(&self) -> bool {
        let mut basis = Vec::new();
        for &b in self.reg.iter().chain(&self.lane).chain(&self.warp) {
            insert(&mut basis, b);
        }
        basis.len() == log2(self.numel()) as usize
    }

    /// A coalescing layout: each thread holds `vec` consecutive elements
    /// of a row, consecutive lanes hold consecutive chunks, warps split the
    /// rows; what is left over repeats in registers.
    pub fn blocked(shape: &[usize], warps: usize, vec: usize) -> Layout {
        let (r, c) = match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            _ => (shape[0], shape[1]),
        };
        let (lc, lr) = (log2(c), log2(r));
        // Position bits, columns first (low), then rows.
        let col_bits: Vec<u32> = (0..lc).map(|i| 1 << i).collect();
        let row_bits: Vec<u32> = (0..lr).map(|i| 1 << (lc + i)).collect();
        let v = log2(vec.min(c)) as usize;
        let mut reg: Vec<u32> = col_bits[..v].to_vec();
        let mut rest_cols: Vec<u32> = col_bits[v..].to_vec();
        let mut rest_rows = row_bits.clone();
        let mut lane = [0u32; 5];
        for l in lane.iter_mut() {
            *l = if !rest_cols.is_empty() {
                rest_cols.remove(0)
            } else if !rest_rows.is_empty() {
                rest_rows.remove(0)
            } else {
                0
            };
        }
        let mut warp = Vec::new();
        for _ in 0..log2(warps) {
            warp.push(if !rest_rows.is_empty() {
                rest_rows.remove(0)
            } else if !rest_cols.is_empty() {
                rest_cols.remove(0)
            } else {
                0
            });
        }
        reg.extend(rest_cols);
        reg.extend(rest_rows);
        Layout {
            shape: shape.to_vec(),
            reg,
            lane,
            warp,
        }
    }

    /// The accumulator of `mma.m16n8k8` tiles over a [BM, BN] tile: warps
    /// in a `wm x wn` grid, each covering BM/wm x BN/wn in m16n8 pieces.
    /// Within a piece, with g = lane / 4 and t = lane % 4, registers 0 and 1
    /// hold (g, 2t) and (g, 2t+1), registers 2 and 3 the same 8 rows lower.
    pub fn mma(shape: &[usize], wm: usize, wn: usize) -> Layout {
        let (bm, bn) = (shape[0], shape[1]);
        let lc = log2(bn);
        let rb = |r: u32| r << lc;
        let (tm, tn) = (bm / wm, bn / wn);
        let mut reg = vec![1, rb(8)];
        let mut x = 8;
        while x < tn {
            reg.push(x as u32);
            x *= 2;
        }
        let mut y = 16;
        while y < tm {
            reg.push(rb(y as u32));
            y *= 2;
        }
        let lane = [2, 4, rb(1), rb(2), rb(4)];
        let mut warp = Vec::new();
        let mut s = tm;
        while s < bm {
            warp.push(rb(s as u32));
            s *= 2;
        }
        let mut s = tn;
        while s < bn {
            warp.push(s as u32);
            s *= 2;
        }
        Layout {
            shape: shape.to_vec(),
            reg,
            lane,
            warp,
        }
    }

    /// The accumulator of a CUDA-core matmul: each thread a `tm x tn`
    /// block of registers (columns first in register order, so register
    /// `k` is column `k % tn`, row `k / tn` of the block); lanes and then
    /// warps tile the rest, columns first.
    pub fn simt(shape: &[usize], warps: usize, tm: usize, tn: usize) -> Layout {
        let (bm, bn) = (shape[0], shape[1]);
        let lc = log2(bn);
        let mut reg: Vec<u32> = (0..log2(tn)).map(|i| 1 << i).collect();
        let mut cols: Vec<u32> = (log2(tn)..lc).map(|i| 1 << i).collect();
        let mut rows: Vec<u32> = (log2(tm)..log2(bm)).map(|i| 1 << (lc + i)).collect();
        let tm_bits: Vec<u32> = (0..log2(tm)).map(|i| 1 << (lc + i)).collect();
        let mut lane = [0u32; 5];
        for l in lane.iter_mut() {
            *l = if !cols.is_empty() {
                cols.remove(0)
            } else if !rows.is_empty() {
                rows.remove(0)
            } else {
                0
            };
        }
        let mut warp = Vec::new();
        for _ in 0..log2(warps) {
            warp.push(if !rows.is_empty() {
                rows.remove(0)
            } else if !cols.is_empty() {
                cols.remove(0)
            } else {
                0
            });
        }
        reg.extend(tm_bits);
        // Whatever is left repeats in registers.
        reg.extend(cols);
        reg.extend(rows);
        Layout {
            shape: shape.to_vec(),
            reg,
            lane,
            warp,
        }
    }

    /// Positions of `self.shape` mapped through `f`, into a layout of
    /// `shape`; register bases that become zero or dependent are dropped.
    fn map(&self, shape: &[usize], f: impl Fn(u32) -> u32) -> Layout {
        let mut reg = Vec::new();
        let mut span = Vec::new();
        for &b in &self.reg {
            let m = f(b);
            if insert(&mut span, m) {
                reg.push(m);
            }
        }
        let lane = self.lane.map(&f);
        Layout {
            shape: shape.to_vec(),
            reg,
            lane,
            warp: self.warp.iter().map(|&b| f(b)).collect(),
        }
    }

    /// Where position `b` of this tile lands in its reduction along `axis`.
    pub fn slice_pos(&self, axis: usize, b: u32) -> u32 {
        let c = self.cols() as u32;
        match (self.rank(), axis) {
            (1, _) => 0,
            (_, 1) => b / c,
            _ => b % c,
        }
    }

    /// The layout of the reduction of this tile along `axis`.
    pub fn slice(&self, axis: usize) -> Layout {
        let shape: Vec<usize> = match (self.rank(), axis) {
            (1, _) => vec![],
            (_, 1) => vec![self.shape[0]],
            _ => vec![self.cols()],
        };
        self.map(&shape, |b| self.slice_pos(axis, b))
    }

    /// Where position `b` of this tile reads from in a tile of `src_shape`
    /// (this shape with some dimensions 1) broadcast to it.
    pub fn project_pos(&self, src_shape: &[usize], b: u32) -> u32 {
        let c = self.cols() as u32;
        let sc = *src_shape.last().unwrap_or(&1) as u32;
        let (keep_r, keep_c) = match (self.rank(), src_shape.len()) {
            (2, 2) => (src_shape[0] != 1, src_shape[1] != 1),
            _ => (false, src_shape.first().copied().unwrap_or(1) != 1),
        };
        let (r, cc) = (b / c, b % c);
        (if keep_r { r } else { 0 }) * sc + if keep_c { cc } else { 0 }
    }

    /// The layout `src_shape` (this shape with some dimensions 1) needs for
    /// a broadcast to this layout to need no data movement.
    pub fn project(&self, src_shape: &[usize]) -> Layout {
        self.map(src_shape, |b| self.project_pos(src_shape, b))
    }

    /// The same distribution viewed with `shape` (equal element count,
    /// equal positions): expand_dims and its inverse.
    pub fn reshape(&self, shape: &[usize]) -> Layout {
        Layout {
            shape: shape.to_vec(),
            ..self.clone()
        }
    }

    /// The transposed tile's layout (data stays where it is).
    pub fn trans(&self) -> Layout {
        let (r, c) = (self.shape[0], self.shape[1]);
        let lc = log2(c);
        let lr = log2(r);
        let f = |b: u32| ((b & (c as u32 - 1)) << lr) | (b >> lc);
        Layout {
            shape: vec![c, r],
            reg: self.reg.iter().map(|&b| f(b)).collect(),
            lane: self.lane.map(f),
            warp: self.warp.iter().map(|&b| f(b)).collect(),
        }
    }

    /// When every thread of `other` holds its elements in the same thread
    /// here: for each register of `other`, the register here holding it.
    pub fn reg_map_from(&self, other: &Layout) -> Option<Vec<usize>> {
        if self.shape.iter().product::<usize>() != other.numel() || self.lane != other.lane || self.warp != other.warp {
            return None;
        }
        let mut at = std::collections::HashMap::new();
        for k in 0..self.nregs() {
            at.entry(self.reg_pos(k)).or_insert(k);
        }
        (0..other.nregs()).map(|k| at.get(&other.reg_pos(k)).copied()).collect()
    }

    /// Lane bits whose lanes hold different positions along `axis` only
    /// (the shuffle partners of a reduction), and likewise warp bits.
    pub fn axis_bits(&self, axis: usize) -> (Vec<usize>, Vec<usize>) {
        let on_axis = |b: u32| -> Option<bool> {
            if b == 0 {
                return None;
            }
            let c = self.cols() as u32;
            let (row, col) = (b / c, b % c);
            let a = if self.rank() == 2 && axis == 0 { row } else { col };
            let o = if self.rank() == 2 && axis == 0 { col } else { row };
            match (a != 0, o != 0) {
                (true, false) => Some(true),
                (false, true) => Some(false),
                _ => panic!("mixed basis {b:#x} in {self:?}"),
            }
        };
        let lanes = (0..5).filter(|&i| on_axis(self.lane[i]) == Some(true)).collect();
        let warps = (0..self.warp.len())
            .filter(|&i| on_axis(self.warp[i]) == Some(true))
            .collect();
        (lanes, warps)
    }

    /// Positions held by register `k` of every thread, for checking.
    pub fn owners(&self) -> Vec<Vec<(usize, usize, usize)>> {
        let mut out = vec![Vec::new(); self.numel()];
        for w in 0..1 << self.warp.len() {
            for l in 0..32 {
                for k in 0..self.nregs() {
                    out[(self.thread_pos(l, w) ^ self.reg_pos(k)) as usize].push((w, l, k));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_covers_and_coalesces() {
        for (shape, w, v) in [
            (vec![64, 32], 4, 4),
            (vec![1024], 4, 4),
            (vec![4, 1024], 4, 4),
            (vec![16], 4, 4),
            (vec![128, 64], 8, 8),
        ] {
            let l = Layout::blocked(&shape, w, v);
            assert!(l.covers(), "{shape:?}");
            // Register bits first walk consecutive columns.
            for i in 0..v.min(l.cols()).trailing_zeros() as usize {
                assert_eq!(l.reg[i], 1 << i);
            }
        }
    }

    #[test]
    fn mma_layout_matches_the_fragment_spec() {
        let l = Layout::mma(&[64, 32], 2, 2);
        assert!(l.covers());
        // Warp 0, lane 5 (g = 1, t = 1): register 0 at (1, 2), 1 at (1, 3),
        // 2 at (9, 2), 3 at (9, 3); register 4 the next n8 piece.
        let at = |k| l.coords(l.thread_pos(5, 0) ^ l.reg_pos(k));
        assert_eq!(
            [at(0), at(1), at(2), at(3), at(4)],
            [(1, 2), (1, 3), (9, 2), (9, 3), (1, 10)]
        );
        // Warp 1 starts 32 rows down (warps split m first).
        assert_eq!(l.coords(l.thread_pos(0, 1)), (32, 0));
        // Each element held exactly once.
        assert!(l.owners().iter().all(|o| o.len() == 1));
    }

    #[test]
    fn slices_and_projections_agree() {
        // A reduction's result broadcast back needs no data movement.
        for l in [
            Layout::mma(&[64, 64], 4, 1),
            Layout::blocked(&[16, 128], 4, 4),
            Layout::simt(&[64, 64], 4, 4, 4),
        ] {
            for axis in [0, 1] {
                let s = l.slice(axis);
                let mut src = l.shape.clone();
                src[axis] = 1;
                let p = l.project(&src);
                assert!(s.reshape(&src).reg_map_from(&p).is_some(), "{l:?} axis {axis}");
                assert!(s.covers());
            }
        }
    }

    #[test]
    fn transpose_moves_no_data() {
        let l = Layout::blocked(&[16, 64], 4, 4);
        let t = l.trans();
        for w in 0..4 {
            for lane in 0..32 {
                for k in 0..l.nregs() {
                    let (r, c) = l.coords(l.thread_pos(lane, w) ^ l.reg_pos(k));
                    assert_eq!(t.coords(t.thread_pos(lane, w) ^ t.reg_pos(k)), (c, r));
                }
            }
        }
    }

    #[test]
    fn simt_registers_form_a_block() {
        let l = Layout::simt(&[64, 64], 4, 4, 4);
        assert!(l.covers());
        let base = l.coords(l.thread_pos(3, 1));
        for k in 0..16 {
            let (r, c) = l.coords(l.thread_pos(3, 1) ^ l.reg_pos(k));
            assert_eq!((r - base.0, c - base.1), (k / 4, k % 4));
        }
    }
}
