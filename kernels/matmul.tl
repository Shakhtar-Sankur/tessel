# C = A B (+ bias), f16 or f32 operands, f32 accumulation.

kernel matmul(A: f16[M, K], B: f16[K, N], C: f16[M, N]):
    meta BM = 128, BN = 128, BK = 32, G = 8
    grid(cdiv(M, BM) * cdiv(N, BN))
    # Programs walk the output in groups of G block-rows, so the blocks
    # running together share rows of A and columns of B in the L2 cache.
    pid = program_id(0)
    width = G * cdiv(N, BN)
    first = (pid // width) * G
    rows = minimum(cdiv(M, BM) - first, G)
    i = (first + (pid % width) % rows) * BM
    j = ((pid % width) // rows) * BN
    acc = zeros([BM, BN], f32)
    for k in range(0, K, BK):
        acc = dot(A[i : +BM, k : +BK], B[k : +BK, j : +BN], acc)
    C[i : +BM, j : +BN] = acc

kernel matmul_f32(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
    meta BM = 64, BN = 64, BK = 16
    grid(cdiv(N, BN), cdiv(M, BM))
    i = program_id(1) * BM
    j = program_id(0) * BN
    acc = zeros([BM, BN], f32)
    for k in range(0, K, BK):
        acc = dot(A[i : +BM, k : +BK], B[k : +BK, j : +BN], acc)
    C[i : +BM, j : +BN] = acc

# A linear layer as a model runs it: Y = X W^T + b, W stored [out, in].
kernel linear(X: f16[M, K], W: f16[N, K], b: f16[N], Y: f16[M, N]):
    meta BM = 64, BN = 128, BK = 32
    grid(cdiv(N, BN), cdiv(M, BM))
    i = program_id(1) * BM
    j = program_id(0) * BN
    acc = zeros([BM, BN], f32)
    for k in range(0, K, BK):
        acc = dot(X[i : +BM, k : +BK], trans(W[j : +BN, k : +BK]), acc)
    Y[i : +BM, j : +BN] = acc + f32(b[j : +BN])[None, :]
