# The kernels of a Llama-family decoder (RMSNorm, rotary embeddings,
# grouped-query attention, SwiGLU), as the engine in src/llm runs them.
#
# Activations are [T, ...] for the T tokens of a step: the residual stream
# in f32, everything a matmul reads in f16. Shapes are compiled in, so the
# engine pads T to a few sizes; padding rows compute garbage nobody reads,
# and their cache slots are out of range, so their writes are dropped.

# X[t] = E[tok[t]]
kernel embed(tok: i32[T], E: f16[V, Dm], X: f32[T, Dm]):
    meta BD = 1024
    grid(T, cdiv(Dm, BD))
    t = program_id(0)
    c = program_id(1) * BD
    row = tok[t]
    X[t, c : +BD] = f32(E[row, c : +BD])

# Y = X / rms(X) * w, rows of the f32 residual stream to f16.
kernel rmsnorm(X: f32[T, Dm], W: f16[Dm], Y: f16[T, Dm], eps: f32):
    meta BR = 1, BC = 2048
    grid(cdiv(T, BR))
    r = program_id(0) * BR
    x = X[r : +BR, 0 : +BC]
    ms = sum(x * x, axis=1) / Dm
    Y[r : +BR, 0 : +BC] = f16(x * rsqrt(ms + eps)[:, None] * f32(W[0 : +BC])[None, :])

# Y = X W^T, W stored [out, in] as checkpoints store it.
kernel linear(X: f16[M, K], W: f16[N, K], Y: f16[M, N]):
    meta BM = 16, BN = 64, BK = 64
    grid(cdiv(N, BN), cdiv(M, BM))
    i = program_id(1) * BM
    j = program_id(0) * BN
    acc = zeros([BM, BN], f32)
    for k in range(0, K, BK):
        acc = dot(X[i : +BM, k : +BK], trans(W[j : +BN, k : +BK]), acc)
    Y[i : +BM, j : +BN] = f16(acc)

# R += X W^T: the output projections, added into the residual stream.
kernel linear_residual(X: f16[M, K], W: f16[N, K], R: f32[M, N]):
    meta BM = 16, BN = 64, BK = 64
    grid(cdiv(N, BN), cdiv(M, BM))
    i = program_id(1) * BM
    j = program_id(0) * BN
    acc = zeros([BM, BN], f32)
    for k in range(0, K, BK):
        acc = dot(X[i : +BM, k : +BK], trans(W[j : +BN, k : +BK]), acc)
    R[i : +BM, j : +BN] = R[i : +BM, j : +BN] + acc

# Logits = X W^T in f32.
kernel logits(X: f16[M, K], W: f16[N, K], Y: f32[M, N]):
    meta BM = 16, BN = 64, BK = 64
    grid(cdiv(N, BN), cdiv(M, BM))
    i = program_id(1) * BM
    j = program_id(0) * BN
    acc = zeros([BM, BN], f32)
    for k in range(0, K, BK):
        acc = dot(X[i : +BM, k : +BK], trans(W[j : +BN, k : +BK]), acc)
    Y[i : +BM, j : +BN] = acc

# SwiGLU's first half, both matmuls in one pass over X:
# Y = silu(X Wg^T) * (X Wu^T).
kernel gate_up(X: f16[M, K], Wg: f16[F, K], Wu: f16[F, K], Y: f16[M, F]):
    meta BM = 16, BN = 64, BK = 64
    grid(cdiv(F, BN), cdiv(M, BM))
    i = program_id(1) * BM
    j = program_id(0) * BN
    g = zeros([BM, BN], f32)
    u = zeros([BM, BN], f32)
    for k in range(0, K, BK):
        x = X[i : +BM, k : +BK]
        g = dot(x, trans(Wg[j : +BN, k : +BK]), g)
        u = dot(x, trans(Wu[j : +BN, k : +BK]), u)
    Y[i : +BM, j : +BN] = f16(g / (1.0 + exp(-g)) * u)

# Rotary embeddings on the queries of QKV ([T, (H + 2G) * D], as the fused
# projection writes it), into Q [T, H, D]. Llama's convention: the first
# and second halves of each head are the pairs.
kernel rope_q(QKV: f16[T, W], pos: i32[T], C: f32[NPOS, D2], S: f32[NPOS, D2], Q: f16[T, H, D]):
    grid(T, H)
    t = program_id(0)
    h = program_id(1)
    p = pos[t]
    c = C[p, 0 : +D2]
    s = S[p, 0 : +D2]
    x1 = f32(QKV[t, h * D : +D2])
    x2 = f32(QKV[t, h * D + D2 : +D2])
    Q[t, h, 0 : +D2] = f16(x1 * c - x2 * s)
    Q[t, h, D2 : +D2] = f16(x2 * c + x1 * s)

# The keys (rotated) and values of QKV into the paged cache, at slot[t]
# (page * P + offset; out of range for padding), and into K, V [T, G, D]
# for the prompt's own attention.
kernel kv_write(QKV: f16[T, W], pos: i32[T], slot: i32[T], C: f32[NPOS, D2], S: f32[NPOS, D2], Kc: f16[NS, G, D], Vc: f16[NS, G, D], K: f16[T, G, D], V: f16[T, G, D]):
    grid(T, G)
    t = program_id(0)
    g = program_id(1)
    p = pos[t]
    at = slot[t]
    c = C[p, 0 : +D2]
    s = S[p, 0 : +D2]
    k0 = W - 2 * G * D + g * D
    x1 = f32(QKV[t, k0 : +D2])
    x2 = f32(QKV[t, k0 + D2 : +D2])
    y1 = f16(x1 * c - x2 * s)
    y2 = f16(x2 * c + x1 * s)
    v = QKV[t, k0 + G * D : +D]
    Kc[at, g, 0 : +D2] = y1
    Kc[at, g, D2 : +D2] = y2
    Vc[at, g, 0 : +D] = v
    K[t, g, 0 : +D2] = y1
    K[t, g, D2 : +D2] = y2
    V[t, g, 0 : +D] = v

# Causal attention over a prompt, grouped-query: query head h reads key and
# value head h // (H // G).
kernel prefill_attention(Q: f16[S, H, D], K: f16[S, G, D], V: f16[S, G, D], starts: i32[S], O: f16[S, H, D], scale: f32):
    meta BM = 64, BN = 64
    grid(cdiv(S, BM), H)
    i = program_id(0) * BM
    h = program_id(1)
    g = h // (H // G)
    q = Q[i : +BM, h, 0 : +D]
    rows = i + arange(BM)
    # Each row attends causally within its own sequence, which begins at
    # row starts[row] of the step: several prompts packed into one step.
    # The block's first row has the earliest start.
    st = starts[i : +BM]
    lo = starts[i]
    # The running maximum starts finite: a row's first key blocks can be
    # wholly masked (before its own prompt begins), and -inf - -inf would
    # make them NaN instead of contributing nothing.
    m = full([BM], -1e30, f32)
    l = zeros([BM], f32)
    acc = zeros([BM, D], f32)
    for j in range(lo, minimum(i + BM, S), BN):
        s = dot(q, trans(K[j : +BN, g, 0 : +D])) * scale
        cols = j + arange(BN)
        s = where((cols[None, :] <= rows[:, None]) & (cols[None, :] >= st[:, None]) & (cols[None, :] < S), s, -inf)
        m2 = maximum(m, max(s, axis=1))
        p = exp(s - m2[:, None])
        a = exp(m - m2)
        l = l * a + sum(p, axis=1)
        acc = acc * a[:, None] + dot(f16(p), V[j : +BN, g, 0 : +D])
        m = m2
    O[i : +BM, h, 0 : +D] = f16(acc / l[:, None])

# One query token per sequence against its paged cache: Q [B, H, D], the
# cache [pages, P, G, D], table[b, n] = the page holding tokens n*P ..
# n*P+P-1 of sequence b, lens[b] its tokens so far (0 for padding).
kernel decode_attention(Q: f16[B, H, D], Kc: f16[NP, P, G, D], Vc: f16[NP, P, G, D], table: i32[B, MP], lens: i32[B], O: f16[B, H, D], scale: f32):
    grid(B, H)
    b = program_id(0)
    h = program_id(1)
    g = h // (H // G)
    n = lens[b]
    q = f32(Q[b, h, 0 : +D])
    m = full([1], -inf, f32)
    l = zeros([1], f32)
    acc = zeros([1, D], f32)
    for t in range(0, n, P):
        page = table[b, t // P]
        k = f32(Kc[page, 0 : +P, g, 0 : +D])
        sc = sum(k * q[None, :], axis=1) * scale
        sc = where(t + arange(P) < n, sc, -inf)
        m2 = maximum(m, max(sc[None, :], axis=1))
        p = exp(sc[None, :] - m2[:, None])
        a = exp(m - m2)
        l = l * a + sum(p, axis=1)
        v = f32(Vc[page, 0 : +P, g, 0 : +D])
        acc = acc * a[:, None] + sum(trans(p) * v, axis=0)[None, :]
        m = m2
    O[b, h, 0 : +D] = f16(sum(acc / l[:, None], axis=0))

# The final norm of the rows whose logits are wanted: Y[b] = rmsnorm(X[idx[b]]).
kernel rmsnorm_rows(X: f32[T, Dm], idx: i32[B], W: f16[Dm], Y: f16[B, Dm], eps: f32):
    meta BC = 2048
    grid(B)
    b = program_id(0)
    r = idx[b]
    x = X[r, 0 : +BC]
    ms = sum(x * x, axis=0) / Dm
    Y[b, 0 : +BC] = f16(x * rsqrt(ms + eps) * f32(W[0 : +BC]))
