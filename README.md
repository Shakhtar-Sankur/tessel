# tessel

A tile language for GPU kernels, and its compiler, written from scratch in
Rust with no dependencies. Kernels are written once, against *tiles*
(small blocks of values), and compiled for each target: today CUDA for
NVIDIA tensor cores; Metal (Apple GPUs) and Pallas (Google TPUs) are next.

```python
kernel matmul(A: f16[M, K], B: f16[K, N], C: f16[M, N]):
    meta BM = 128, BN = 128, BK = 32
    grid(cdiv(N, BN), cdiv(M, BM))
    i = program_id(1) * BM
    j = program_id(0) * BN
    acc = zeros([BM, BN], f32)
    for k in range(0, K, BK):
        acc = dot(A[i : +BM, k : +BK], B[k : +BK, j : +BN], acc)
    C[i : +BM, j : +BN] = acc
```

The compiler gives every tile a *linear layout* (which thread holds which
element in which register, as in Triton), picks tensor-core fragment
layouts for matmuls, coalescing layouts for loads and slices of them for
reductions, recomputes index arithmetic in place instead of moving it, and
generates CUDA with `ldmatrix`/`mma.sync`, 16-byte packed staging and
software-pipelined loads. The same generated source also compiles as C++
against an emulator of the CUDA execution model, so every kernel is tested
without a GPU.

Work in progress: benchmarks against cuBLAS, PyTorch and Triton on a T4
come next. Kernels so far: `kernels/` (matmul, FlashAttention, paged
decode attention, softmax, RMSNorm).

## Usage

```sh
cargo build --release
./target/release/tessel run attention flash_attention --shapes 2x100x32,2x100x32,2x100x32,2x100x32
./target/release/tessel cuda matmul matmul --shapes 2048x2048,2048x2048,2048x2048
./target/release/tessel bench            # on an NVIDIA GPU
bash scripts/colab.sh                     # in Colab: tests, benchmarks, report
```

## License

Apache-2.0
