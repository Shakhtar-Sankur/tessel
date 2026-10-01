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

Run 3 (commit c7ab694, a Kaggle T4, 4 rounds; raw rows in `bench/t4/`).
tessel's speed relative to the fastest other engine timed in the same
round, as the median of the rounds (above 1 is faster):

| Case | tessel, ms | Fastest other, ms | tessel's speed | Rounds |
|---|---|---|---|---|
| GEMM 1024³, fp16, fp32 accumulation | 0.148 | cuBLAS 0.148 | **1.00x** | 0.58, 1.08, 1.14, 0.93 |
| GEMM 2048³ | 0.985 | cuBLAS 0.941 | **1.00x** | 1.04, 0.93, 1.02, 0.98 |
| GEMM 4096³ | 7.381 | cuBLAS 7.357 | **1.04x** | 1.02, 0.94, 1.08, 1.06 |
| Causal attention, 32 heads, D 64, S 512 | 0.146 | SDPA 0.276 | **1.90x** | 1.11, 1.95, 1.91, 1.88 |
| S 1024 | 0.405 | SDPA 0.526 | **1.28x** | 0.88, 1.30, 1.27, 1.35 |
| S 2048 | 1.247 | SDPA 1.642 | **1.36x** | 1.45, 1.36, 0.98, 1.37 |
| S 4096 | 4.416 | SDPA 6.362 | **1.44x** | 1.61, 1.44, 1.44, 1.44 |
| Softmax 4096x4096, fp32 | 0.583 (230 GB/s) | Triton 0.628 | **1.07x** | 1.06, 1.06, 1.07, 1.12 |
| RMSNorm 4096x4096, fp16 | 0.291 (231 GB/s) | torch.compile 0.368 | **1.27x** | 1.24, 1.26, 1.29, 1.27 |

Times are medians of the rounds' medians (100 launches each); every
output is checked against a float64 reference, and tessel's errors match
the baselines' (3e-4 relative for fp16 GEMM and attention). cuBLAS is
`torch.matmul` with fp16 reduction disabled (fp32 accumulation, as
tessel); SDPA is `torch.nn.functional.scaled_dot_product_attention`
(PyTorch 2.10). Triton 3.6 does not use tensor cores on the T4 (its
matmul PTX has no `mma`), so it is far behind on GEMM and attention there
(152 ms at 4096³) and is left out of the table.

How to read it: the T4 throttles as it heats, so the rounds alternate
which side runs first and the table compares each round's numbers with
each other. tessel's first round is low where it also tunes (compiling
between timings lets the clocks fall). Run 2, before the GEMM operands
were XOR-swizzled in shared memory (so two 128x128 blocks fit on an SM),
had GEMM at 0.74x and 0.71x of cuBLAS at 1024³ and 2048³.

## An LLM engine on tessel kernels

`src/llm` runs Llama-family models (Llama, TinyLlama, Mistral-style
grouped-query attention) with every GPU operation a tessel kernel from
`kernels/llm.tl`: embedding, RMSNorm, the fused QKV, output and MLP
projections (gate and up in one pass, SiLU fused), rotary embeddings,
writes into a paged KV cache, causal prefill attention and paged decode
attention, both grouped-query, and the LM head over only the rows that
need logits. A scheduler batches continuously: sequences join as cache
pages allow, every step decodes one token for each, and a finished
sequence frees its pages at once. Steps are padded to a few sizes, so a
handful of compilations serve every step. Checkpoints load from Hugging
Face safetensors (F16, BF16 or F32).

On the emulator, a small random Llama matches a plain f32 reference
model: logits after a prompt within 3e-4 (relative to the largest), and
all 19 tokens of a three-sequence continuous-batched generation the
reference's own greedy choices (`tests/llm.rs`). That reference model, on
TinyLlama-1.1B-Chat's real weights loaded by tessel's safetensors reader,
matches Hugging Face transformers' fp32 logits after a chat prompt within
4e-5 (2e-6 of the largest logit), with the same top five tokens
(`examples/reference_logits.rs`). NVRTC compiles every
kernel at TinyLlama-1.1B's shapes without register spills.

On a T4 (Kaggle, commit 2fd72a1; raw rows in `bench/t4/`), TinyLlama-1.1B-Chat
in fp16, greedy, against Hugging Face transformers 5.0 (`generate`, fp16,
the same prompts):

| | tessel | transformers | |
|---|---|---|---|
| Logits after a chat prompt, vs transformers in fp32 | within 3.9e-4 of the largest; top 5 the same | | |
| Greedy generations, 4 prompts, up to 128 tokens | identical to transformers', token for token, all 4 | | |
| 1 sequence at a time (4 requests) | **85.0 tokens/s** | 32.6 tokens/s | 2.6x |
| Batches of 8 (32 requests) | **554 tokens/s** | 247 tokens/s | 2.2x |
| Batches of 32 (32 requests) | **1169 tokens/s** | 970 tokens/s | 1.2x |

Tokens per second count each engine's own generated tokens over the whole
run, prompts included (transformers' padded batches end a few tokens
differently: 3843 against tessel's 3853 at batch 8). transformers'
`generate` is eager PyTorch, the reference implementation rather than a
serving engine; vLLM and llama.cpp are the next comparisons. One sequence
at 85 tokens/s is 11.8 ms a token, against 6.9 ms to read the weights once
at the T4's 320 GB/s.

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
./target/release/tessel llm-tiny /tmp/tiny && echo '[[1, 5, 9]]' > /tmp/p.json
./target/release/tessel llm /tmp/tiny --prompts /tmp/p.json --device emu --max-tokens 16 --pages 64
bash scripts/llm_colab.sh                 # Kaggle or Colab (T4): TinyLlama-1.1B against transformers
```

## License

Apache-2.0
