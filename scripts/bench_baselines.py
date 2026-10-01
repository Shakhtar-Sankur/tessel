"""The benchmark suite's baselines, on the same GPU and sizes as `tessel bench`:

- GEMM, fp16 in and out, fp32 accumulation: torch.matmul (cuBLAS), with
  reduced-precision reductions disabled so it accumulates in fp32 as tessel
  does; and Triton's matmul (the tutorial kernel, autotuned).
- Causal attention, fp16: torch's scaled_dot_product_attention (on a T4 the
  memory-efficient CUTLASS kernel), and a Triton FlashAttention kernel.
- Softmax (fp32) and RMSNorm (fp16): torch eager, torch.compile, and Triton
  (the tutorial softmax).

Every engine's output is compared with a float64 reference on sampled
elements. One JSON line per (case, engine), like `tessel bench`.

Usage: python scripts/bench_baselines.py [--quick] [--iters 30] [--json OUT]
"""

import argparse
import glob
import json
import time
import math
import os
import shutil
import sys

# Triton's compiled kernels land here, so we can check what they use.
TRITON_CACHE = "/tmp/tessel_triton_cache"
shutil.rmtree(TRITON_CACHE, ignore_errors=True)
os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None


def _nvml():
    """The SM clock reader of NVML (device 0), or None."""
    try:
        import ctypes

        lib = ctypes.CDLL("libnvidia-ml.so.1")
        dev = ctypes.c_void_p()
        if lib.nvmlInit_v2() != 0 or lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev)) != 0:
            return None

        def read():
            v = ctypes.c_uint()
            return v.value if lib.nvmlDeviceGetClockInfo(dev, 1, ctypes.byref(v)) == 0 else None

        return read
    except Exception:
        return None


SM_MHZ = _nvml()
LAST_MHZ = None


def bench(fn, iters):
    # Warm up for at least 200 ms of launches (and 3), so the clocks have
    # ramped up after any idle time; tessel's timing does the same.
    t, n = time.perf_counter(), 0
    while n < 3 or time.perf_counter() - t < 0.2:
        fn()
        n += 1
        if n % 8 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    # The clock the timed launches ran at (read before the GPU idles down).
    global LAST_MHZ
    LAST_MHZ = SM_MHZ() if SM_MHZ else None
    ts.sort()
    return ts[len(ts) // 2], ts[0]


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BM": 128, "BN": 128, "BK": 32, "G": 8}, num_stages=2, num_warps=4),
            triton.Config({"BM": 128, "BN": 128, "BK": 32, "G": 8}, num_stages=2, num_warps=8),
            triton.Config({"BM": 128, "BN": 64, "BK": 32, "G": 8}, num_stages=2, num_warps=4),
            triton.Config({"BM": 64, "BN": 128, "BK": 32, "G": 8}, num_stages=2, num_warps=4),
            triton.Config({"BM": 64, "BN": 64, "BK": 32, "G": 8}, num_stages=2, num_warps=4),
            triton.Config({"BM": 128, "BN": 256, "BK": 32, "G": 8}, num_stages=2, num_warps=8),
            triton.Config({"BM": 64, "BN": 64, "BK": 64, "G": 8}, num_stages=2, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        width = G * npn
        first = (pid // width) * G
        rows = tl.minimum(npm - first, G)
        pm = first + (pid % width) % rows
        pn = (pid % width) // rows
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            x = tl.load(a + rm[:, None] * K + (k * BK + rk)[None, :], mask=(rm[:, None] < M) & ((k * BK + rk)[None, :] < K), other=0.0)
            y = tl.load(b + (k * BK + rk)[:, None] * N + rn[None, :], mask=((k * BK + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc = tl.dot(x, y, acc)
        tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))

    def triton_matmul(a, b):
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        _matmul[lambda m: (triton.cdiv(M, m["BM"]) * triton.cdiv(N, m["BN"]),)](a, b, c, M, N, K)
        return c

    @triton.jit
    def _attn(Q, Kc, V, O, scale, S, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        i = tl.program_id(0) * BM
        h = tl.program_id(1)
        base = h * S * D
        rows = i + tl.arange(0, BM)
        d = tl.arange(0, D)
        q = tl.load(Q + base + rows[:, None] * D + d[None, :], mask=rows[:, None] < S, other=0.0)
        m = tl.full((BM,), float("-inf"), tl.float32)
        l = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        for j in range(0, tl.minimum(i + BM, S), BN):
            cols = j + tl.arange(0, BN)
            k = tl.load(Kc + base + cols[:, None] * D + d[None, :], mask=cols[:, None] < S, other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where((cols[None, :] <= rows[:, None]) & (cols[None, :] < S), s, float("-inf"))
            m2 = tl.maximum(m, tl.max(s, 1))
            p = tl.exp(s - m2[:, None])
            a = tl.exp(m - m2)
            l = l * a + tl.sum(p, 1)
            v = tl.load(V + base + cols[:, None] * D + d[None, :], mask=cols[:, None] < S, other=0.0)
            acc = acc * a[:, None] + tl.dot(p.to(tl.float16), v)
            m = m2
        tl.store(O + base + rows[:, None] * D + d[None, :], (acc / l[:, None]).to(tl.float16), mask=rows[:, None] < S)

    def triton_attention(q, k, v, scale):
        """The fastest of a few tile shapes, as a function returning the output."""
        H, S, D = q.shape
        o = torch.empty_like(q)
        best = None
        for bm, bn, w in ((64, 64, 4), (128, 64, 8), (64, 32, 4), (128, 32, 8)):
            run = lambda bm=bm, bn=bn, w=w: _attn[(triton.cdiv(S, bm), H)](q, k, v, o, scale, S, D, bm, bn, num_warps=w, num_stages=2)
            try:
                t = bench(run, 10)[0]
            except Exception:  # noqa: BLE001
                continue
            if best is None or t < best[0]:
                best = (t, run)
        run = best[1]

        def f():
            run()
            return o

        return f

    @triton.jit
    def _softmax(X, Y, C, BC: tl.constexpr):
        r = tl.program_id(0)
        c = tl.arange(0, BC)
        x = tl.load(X + r * C + c, mask=c < C, other=float("-inf"))
        e = tl.exp(x - tl.max(x, 0))
        tl.store(Y + r * C + c, e / tl.sum(e, 0), mask=c < C)

    def triton_softmax(x):
        R, C = x.shape
        y = torch.empty_like(x)
        _softmax[(R,)](x, y, C, triton.next_power_of_2(C), num_warps=8)
        return y


def rel(err, scale):
    return float(err / max(scale, 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json")
    a = ap.parse_args()
    dev = torch.cuda.get_device_name(0)
    out = open(a.json, "a") if a.json else None
    g = torch.Generator(device="cuda").manual_seed(0)

    def emit(kind, label, engine, med, mn, err, flops, nbytes, note=""):
        r = {"kind": kind, "label": label, "engine": engine, "device": dev, "median_ms": round(med, 4), "min_ms": round(mn, 4),
             "tflops": round(flops / (med * 1e-3) / 1e12, 3) if flops else 0.0, "gbps": round(nbytes / (med * 1e-3) / 1e9, 1),
             "rel_err": err, "sm_mhz": LAST_MHZ, "config": note}
        line = json.dumps(r)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()

    def skip(kind, label, engine, e):
        print(json.dumps({"kind": kind, "label": label, "engine": engine, "error": str(e).splitlines()[0][:200]}), flush=True)

    # Warm up, so timings start at the clock the GPU sustains.
    w = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    import time
    t = time.time()
    while time.time() - t < 1.5:
        for _ in range(20):
            torch.matmul(w, w)
        torch.cuda.synchronize()

    def uni(*shape, dtype=torch.float16, scale=1.0):
        return ((torch.rand(*shape, device="cuda", generator=g) * 2 - 1) * scale).to(dtype)

    # GEMM
    for n in ([1024] if a.quick else [1024, 2048, 4096]):
        A, B = uni(n, n), uni(n, n)
        ii = torch.randint(0, n, (256,), device="cuda", generator=g)
        jj = torch.randint(0, n, (256,), device="cuda", generator=g)
        ref = (A.double()[ii] * B.double()[:, jj].T).sum(1)
        label = f"{n}x{n}x{n}"
        engines = [("cublas", lambda: torch.matmul(A, B))]
        if triton is not None:
            engines.append(("triton", lambda: triton_matmul(A, B)))
        for name, f in engines:
            try:
                c = f()
                err = rel((c.double()[ii, jj] - ref).abs().max().item(), ref.abs().max().item())
                med, mn = bench(f, a.iters)
                emit("gemm", label, name, med, mn, err, 2.0 * n**3, 2.0 * 3 * n * n)
            except Exception as e:  # noqa: BLE001
                skip("gemm", label, name, e)

    # Causal attention [H, S, D]
    for s in ([1024] if a.quick else [512, 1024, 2048, 4096]):
        H, D = 32, 64
        q, k, v = uni(H, s, D), uni(H, s, D), uni(H, s, D)
        scale = 1.0 / math.sqrt(D)
        hs = torch.randint(0, H, (4,), generator=g, device="cuda").tolist()
        rs = torch.randint(0, s, (4,), generator=g, device="cuda").tolist()

        def check(o):
            err, sc = 0.0, 0.0
            for h, i in zip(hs, rs):
                sco = (k[h, : i + 1].double() @ q[h, i].double()) * scale
                p = torch.softmax(sco, 0)
                want = p @ v[h, : i + 1].double()
                err = max(err, (o[h, i].double() - want).abs().max().item())
                sc = max(sc, want.abs().max().item())
            return rel(err, sc)

        label = f"H{H} S{s} D{D} causal"
        flops = 2.0 * 2.0 * H * s * s * D / 2.0
        engines = [("torch-sdpa", lambda: F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0])]
        if triton is not None:
            try:
                engines.append(("triton", triton_attention(q, k, v, scale)))
            except Exception as e:  # noqa: BLE001
                skip("attention", label, "triton", e)
        for name, f in engines:
            try:
                o = f()
                med, mn = bench(f, a.iters)
                emit("attention", label, name, med, mn, check(o), flops, 2.0 * 4 * H * s * D)
            except Exception as e:  # noqa: BLE001
                skip("attention", label, name, e)

    # Softmax, f32
    R, C = 4096, 4096
    x = uni(R, C, dtype=torch.float32, scale=4.0)
    ref = torch.softmax(x[::97].double(), 1)
    engines = [("torch", lambda: torch.softmax(x, 1)), ("torch.compile", torch.compile(lambda: torch.softmax(x, 1)))]
    if triton is not None:
        engines.append(("triton", lambda: triton_softmax(x)))
    for name, f in engines:
        try:
            y = f()
            med, mn = bench(f, a.iters)
            emit("softmax", f"{R}x{C} f32", name, med, mn, (y[::97].double() - ref).abs().max().item(), 0, 8.0 * R * C)
        except Exception as e:  # noqa: BLE001
            skip("softmax", f"{R}x{C} f32", name, e)

    # RMSNorm, f16 (computed in f32)
    x = uni(R, C, scale=2.0)
    w = uni(C)

    def rms():
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(1, keepdim=True) + 1e-5) * w.float()).half()

    xr = x[::97].double()
    ref = xr * torch.rsqrt(xr.pow(2).mean(1, keepdim=True) + 1e-5) * w.double()
    for name, f in (("torch", rms), ("torch.compile", torch.compile(rms))):
        try:
            y = f()
            med, mn = bench(f, a.iters)
            err = rel((y[::97].double() - ref).abs().max().item(), ref.abs().max().item())
            emit("rmsnorm", f"{R}x{C} f16", name, med, mn, err, 0, 4.0 * R * C)
        except Exception as e:  # noqa: BLE001
            skip("rmsnorm", f"{R}x{C} f16", name, e)


    # Does Triton's matmul use tensor cores on this GPU? (mma instructions
    # in the PTX it generated.)
    if triton is not None:
        ptx = [p for p in glob.glob(f"{TRITON_CACHE}/**/*.ptx", recursive=True) if "_matmul" in os.path.basename(p)]
        uses = any("mma." in open(p).read() for p in ptx)
        note = {"kind": "note", "engine": "triton", "triton_version": triton.__version__, "matmul_ptx_files": len(ptx),
                "matmul_uses_mma": uses}
        print(json.dumps(note), flush=True)
        if out:
            out.write(json.dumps(note) + "\n")


if __name__ == "__main__":
    sys.exit(main())
