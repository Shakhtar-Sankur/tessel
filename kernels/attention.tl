# FlashAttention, forward, causal: one program per block of BM queries of
# one head. Scores never leave the chip: each block of BN keys updates a
# running maximum m, a running sum l and the output accumulator.
kernel flash_attention(Q: f16[H, S, D], K: f16[H, S, D], V: f16[H, S, D], O: f16[H, S, D], scale: f32):
    meta BM = 64, BN = 64
    grid(cdiv(S, BM), H)
    i = program_id(0) * BM
    h = program_id(1)
    q = Q[h, i : +BM, 0 : +D]
    rows = i + arange(BM)
    m = full([BM], -inf, f32)
    l = zeros([BM], f32)
    acc = zeros([BM, D], f32)
    for j in range(0, minimum(i + BM, S), BN):
        s = dot(q, trans(K[h, j : +BN, 0 : +D])) * scale
        cols = j + arange(BN)
        s = where((cols[None, :] <= rows[:, None]) & (cols[None, :] < S), s, -inf)
        m2 = maximum(m, max(s, axis=1))
        p = exp(s - m2[:, None])
        a = exp(m - m2)
        l = l * a + sum(p, axis=1)
        acc = acc * a[:, None] + dot(f16(p), V[h, j : +BN, 0 : +D])
        m = m2
    O[h, i : +BM, 0 : +D] = acc / l[:, None]

# Decode attention over a paged KV cache: one program per (sequence, head);
# the cache is a pool of pages of P tokens, and table[s, n] is the page
# holding tokens n*P .. n*P+P-1 of sequence s.
kernel paged_attention(q: f16[B, H, D], Kc: f16[NP, P, H, D], Vc: f16[NP, P, H, D], table: i32[B, MP], lens: i32[B], out: f16[B, H, D], scale: f32):
    meta BQ = 16
    grid(B, H)
    s = program_id(0)
    h = program_id(1)
    n = lens[s]
    qv = f32(q[s, h, 0 : +D])
    m = full([1], -inf, f32)
    l = zeros([1], f32)
    acc = zeros([1, D], f32)
    for t in range(0, n, P):
        page = table[s, t // P]
        k = f32(Kc[page, 0 : +P, h, 0 : +D])
        sc = sum(k * qv[None, :], axis=1) * scale
        sc = where(t + arange(P) < n, sc, -inf)
        m2 = maximum(m, max(sc[None, :], axis=1))
        p = exp(sc[None, :] - m2[:, None])
        a = exp(m - m2)
        l = l * a + sum(p, axis=1)
        v = f32(Vc[page, 0 : +P, h, 0 : +D])
        acc = acc * a[:, None] + sum(trans(p) * v, axis=0)[None, :]
        m = m2
    out[s, h, 0 : +D] = sum(acc / l[:, None], axis=0)
