# tessel

A tile language for GPU and TPU kernels, and its compiler, written from
scratch in Rust with no dependencies. Kernels are written once, against
*tiles* (small blocks of values), and compiled for each target: CUDA for
NVIDIA tensor cores, and Pallas for Google TPUs; Metal (Apple GPUs) is
next.

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

For TPUs, tessel emits a Pallas kernel (a Python module): arrays stay in
HBM, each block access is a DMA to a VMEM buffer, block-table lookups read
SMEM, `dot` runs on the MXU with f32 accumulation, and f16 becomes
bfloat16. A DMA must stay inside its array, so the compiler's range
analysis has to prove every block in bounds, or it refuses the kernel;
indices computed from data (page numbers) are clamped and the access
masked. CI runs every kernel in Pallas's TPU interpret mode against NumPy
and lowers it through Mosaic for TPU v5e; `scripts/colab_tpu.sh` runs
them on a Colab TPU against XLA.

Kernels so far: `kernels/` (matmul, FlashAttention, paged decode
attention, softmax, RMSNorm).

## Results on a Tesla T4

Run 2 (commit 3966b0e, Colab T4, 3 rounds; raw rows in `bench/t4/`).
tessel's speed relative to the fastest other engine timed in the same
round, as the median of the rounds (above 1 is faster):

| Case | tessel, ms | Fastest other, ms | tessel's speed | Rounds |
|---|---|---|---|---|
| GEMM 1024³, fp16, fp32 accumulation | 0.106 | cuBLAS 0.081 | 0.74x | 0.33, 1.07, 0.74 |
| GEMM 2048³ | 1.059 | cuBLAS 0.756 | 0.71x | 0.39, 0.71, 0.87 |
| GEMM 4096³ | 7.676 | cuBLAS 7.649 | **1.01x** | 1.01, 1.00, 1.03 |
| Causal attention, 32 heads, D 64, S 512 | 0.101 | SDPA 0.279 | **2.69x** | 1.33, 2.69, 3.33 |
| S 1024 | 0.280 | SDPA 0.411 | **1.51x** | 0.63, 1.51, 1.75 |
| S 2048 | 1.440 | SDPA 1.732 | **1.18x** | 0.74, 1.24, 1.18 |
| S 4096 | 4.870 | SDPA 6.554 | **1.35x** | 1.31, 1.35, 1.35 |
| Softmax 4096x4096, fp32 | 0.588 (228 GB/s) | Triton 0.661 | **1.13x** | 1.12, 1.13, 1.13 |
| RMSNorm 4096x4096, fp16 | 0.295 (228 GB/s) | torch.compile 0.429 | **1.46x** | 1.51, 1.46, 1.43 |

Times are medians of the rounds' medians (100 launches each); every
output is checked against a float64 reference, and tessel's errors match
the baselines' (3e-4 relative for fp16 GEMM and attention). SDPA is
`torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.11).
Triton 3.6 does not use tensor cores on the T4 (its matmul PTX has no
`mma`), so it is far behind on GEMM and attention there (152 ms at
4096³) and is left out of the table.

How to read it: the T4 throttles as it heats (cuBLAS's 2048³ took 0.55,
0.76 and 0.89 ms in rounds 1 to 3), and in this run tessel always ran
first and tuned in round 1 on a GPU still at idle clocks, which is its
low first round. Later commits alternate the order between rounds, warm
both sides up the same way and log the clocks while each side runs; they
also stage GEMM operands with an XOR swizzle instead of padding, so two
128x128 blocks fit on an SM. Those have not been measured on a GPU yet.

## Usage

```sh
cargo build --release
./target/release/tessel run attention flash_attention --shapes 2x100x32,2x100x32,2x100x32,2x100x32
./target/release/tessel cuda matmul matmul --shapes 2048x2048,2048x2048,2048x2048
./target/release/tessel pallas attention flash_attention --shapes 2x256x64,2x256x64,2x256x64,2x256x64
./target/release/tessel bench            # on an NVIDIA GPU
bash scripts/colab.sh                     # Colab or Kaggle (T4): tests, benchmarks, report
python3 scripts/pallas_check.py --lower   # TPU kernels, interpret mode (needs jax)
bash scripts/colab_tpu.sh                 # in Colab (TPU): the same on a TPU, and against XLA
```

## License

Apache-2.0
