"""Runs tessel's TPU (Pallas) kernels and checks them against NumPy.

Each kernel is generated with `tessel pallas`, imported, and run on random
inputs. Without a TPU, Pallas's TPU interpret mode runs it on the CPU,
modelling the TPU's memory spaces and DMAs (out-of-bounds copies are
errors there). With --lower, each kernel is also lowered for a TPU chip
(default v5e) through Mosaic, Pallas's TPU compiler front end, which needs
no TPU; libtpu's final compile runs only on a TPU (--tpu). References are computed in float64 from the same inputs,
rounded to the kernel's dtypes; TPUs use bfloat16 where the kernel says f16.

usage: python scripts/pallas_check.py [--tessel target/release/tessel] [--lower [CHIP]] [--tpu]
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np


def generate(tessel, file, kernel, shapes, meta=None):
    cmd = [tessel, "pallas", file, kernel, "--shapes", ",".join("x".join(map(str, s)) for s in shapes)]
    if meta:
        cmd += ["--meta", ",".join(f"{k}={v}" for k, v in meta.items())]
    src = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    path = os.path.join(tempfile.mkdtemp(), f"{kernel}.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(kernel, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bf16(x):
    """x rounded to bfloat16, as float64."""
    return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32), np.float64)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def cases(rng):
    """(file, kernel, args, meta, which output, reference, tolerance)."""
    n = lambda *s: rng.standard_normal(s)

    x, y = n(4096), n(4096)
    yield "basic", "add", [x, y, np.zeros(4096)], None, 2, x + y, 1e-6

    x = n(64, 1024) * 3
    yield "basic", "softmax", [x, np.zeros_like(x)], {"BR": 8}, 1, softmax(x), 1e-6

    x, w = bf16(n(64, 1024)), bf16(n(1024))
    ref = x / np.sqrt((x * x).mean(1, keepdims=True) + 1e-5) * w
    yield "basic", "rmsnorm", [x, w, np.zeros_like(x), 1e-5], None, 2, ref, 1e-2

    a, b = bf16(n(512, 256)), bf16(n(256, 384))
    yield "matmul", "matmul", [a, b, np.zeros((512, 384))], {"G": 2}, 2, a @ b, 1e-2

    a, b = n(128, 96), n(96, 192)
    meta = {"BM": 64, "BN": 64, "BK": 32}
    yield "matmul", "matmul_f32", [a, b, np.zeros((128, 192))], meta, 2, a @ b, 1e-4

    x, w, bias = bf16(n(128, 256)), bf16(n(256, 256)), bf16(n(256))
    yield "matmul", "linear", [x, w, bias, np.zeros((128, 256))], None, 3, x @ w.T + bias, 1e-2

    h, s, d = 2, 256, 64
    q, k, v = bf16(n(h, s, d)), bf16(n(h, s, d)), bf16(n(h, s, d))
    scale = d**-0.5
    sc = np.einsum("hqd,hkd->hqk", q, k) * scale
    sc = np.where(np.tril(np.ones((s, s), bool)), sc, -np.inf)
    ref = np.einsum("hqk,hkd->hqd", softmax(sc), v)
    yield "attention", "flash_attention", [q, k, v, np.zeros_like(q), scale], None, 3, ref, 2e-2

    b, h, d, pages, p, mp = 3, 4, 64, 12, 16, 4
    q = bf16(n(b, h, d))
    kc, vc = bf16(n(pages, p, h, d)), bf16(n(pages, p, h, d))
    table = rng.permutation(pages).reshape(b, mp).astype(np.int32)
    lens = np.array([5, 64, 37], np.int32)
    ref = np.zeros((b, h, d))
    for i in range(b):
        ks = np.concatenate([kc[t] for t in table[i]])[: lens[i]]
        vs = np.concatenate([vc[t] for t in table[i]])[: lens[i]]
        ref[i] = np.einsum("hk,khd->hd", softmax(np.einsum("hd,khd->hk", q[i], ks) * scale), vs)
    args = [q, kc, vc, table, lens, np.zeros_like(q), scale]
    yield "attention", "paged_attention", args, None, 5, ref, 2e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tessel", default="target/release/tessel")
    ap.add_argument("--tpu", action="store_true", help="run on a TPU instead of interpret mode")
    ap.add_argument("--lower", nargs="?", const="TPU v5 lite", help="also lower for this TPU (device kind)")
    a = ap.parse_args()
    if a.lower:
        # Mosaic reads the chip's tiling from the current device; name one.
        from jax._src.pallas.mosaic import tpu_info

        tpu_info.get_device_kind = lambda: a.lower
        tpu_info.get_num_device_cores = lambda: 1
    print(f"jax {jax.__version__}, {'TPU' if a.tpu else 'TPU interpret mode on ' + jax.default_backend()}")
    rng = np.random.default_rng(0)
    failed = 0
    for file, kernel, args, meta, out, ref, tol in cases(rng):
        shapes = [np.shape(x) for x in args if np.ndim(x) > 0]
        mod = generate(a.tessel, file, kernel, shapes, meta)
        try:
            res = mod.run(*args, interpret=not a.tpu)
            got = np.asarray(res[mod.OUTPUTS.index(out)].astype(jnp.float32), np.float64)
        except Exception as e:
            failed += 1
            print(f"{kernel:16s} FAIL  {type(e).__name__}: {str(e)[:3000]}")
            continue
        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
        ok = err <= tol
        note = ""
        if a.lower:
            arrs = [np.asarray(x) if np.ndim(x) else x for x in args]
            try:
                exp = jax.export.export(jax.jit(mod.run), platforms=["tpu"])(*arrs)
                assert "tpu_custom_call" in exp.mlir_module()
                note = f"  lowered for {a.lower}"
            except Exception as e:
                ok, note = False, f"  lowering failed: {type(e).__name__}: {str(e)[:2000]}"
        failed += not ok
        print(f"{kernel:16s} grid {mod.GRID}  max error {err:.2e} of max |ref|  {'ok' if ok else 'FAIL'}{note}")
    if failed:
        sys.exit(f"{failed} kernels failed")


if __name__ == "__main__":
    main()
