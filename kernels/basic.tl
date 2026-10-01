# Elementwise, row-wise and reduction kernels.

kernel add(X: f32[N], Y: f32[N], Z: f32[N]):
    meta B = 1024
    grid(cdiv(N, B))
    i = program_id(0) * B
    Z[i : +B] = X[i : +B] + Y[i : +B]

# Softmax over each row of X, one block of rows per program.
kernel softmax(X: f32[R, C], Y: f32[R, C]):
    meta BR = 4, BC = 1024
    grid(cdiv(R, BR))
    r = program_id(0) * BR
    x = load(X[r : +BR, 0 : +BC], other=-inf)
    m = max(x, axis=1)
    e = exp(x - m[:, None])
    Y[r : +BR, 0 : +BC] = e / sum(e, axis=1)[:, None]

# RMSNorm: y = x / sqrt(mean(x^2) + eps) * w, in f32, for f16 rows.
kernel rmsnorm(X: f16[R, C], W: f16[C], Y: f16[R, C], eps: f32):
    meta BR = 4, BC = 1024
    grid(cdiv(R, BR))
    r = program_id(0) * BR
    x = f32(X[r : +BR, 0 : +BC])
    ms = sum(x * x, axis=1) / C
    Y[r : +BR, 0 : +BC] = x * rsqrt(ms + eps)[:, None] * f32(W[0 : +BC])[None, :]
