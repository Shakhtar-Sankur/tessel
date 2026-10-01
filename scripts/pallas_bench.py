"""Times tessel's TPU kernels against XLA on a TPU.

Each case runs tessel's generated Pallas kernel and XLA's own version of
the same computation (jnp.dot; jax.nn.dot_product_attention) on the same
bf16 inputs, checks they agree, and prints one JSON line per case with the
median and minimum over --iters timed runs after warmup.

usage: python scripts/pallas_bench.py [--tessel target/release/tessel] [--iters 50]
       [--interpret]  (small sizes in TPU interpret mode: a smoke test without a TPU)
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from pallas_check import generate


def timed(f, args, iters):
    f = jax.jit(f)
    out = jax.block_until_ready(f(*args))
    for _ in range(3):
        jax.block_until_ready(f(*args))
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t)
    return out, float(np.median(ts)) * 1e3, float(np.min(ts)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tessel", default="target/release/tessel")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--interpret", action="store_true")
    a = ap.parse_args()
    small = a.interpret
    dev = jax.devices()[0]
    key = jax.random.PRNGKey(0)
    cases = []
    for n in (512,) if small else (1024, 2048, 4096):
        ka, kb = jax.random.split(jax.random.fold_in(key, n))
        x = jax.random.normal(ka, (n, n), jnp.bfloat16)
        y = jax.random.normal(kb, (n, n), jnp.bfloat16)
        mod = generate(a.tessel, "matmul", "matmul", [(n, n)] * 3, {"BM": 256, "BN": 256, "BK": 256, "G": 8})
        z = jnp.zeros((n, n), jnp.bfloat16)
        ours = lambda x, y, z, mod=mod: mod.run(x, y, z, interpret=small)[0]
        xla = lambda x, y: jnp.dot(x, y, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        cases.append((f"gemm {n}", ours, (x, y, z), xla, (x, y), 2 * n**3))
    h, d = (2 if small else 32), 64
    for s in (256,) if small else (1024, 2048, 4096):
        ks = jax.random.split(jax.random.fold_in(key, s), 3)
        q, k, v = (jax.random.normal(kk, (h, s, d), jnp.bfloat16) for kk in ks)
        mod = generate(a.tessel, "attention", "flash_attention", [(h, s, d)] * 4, {"BM": 128, "BN": 128})
        scale = d**-0.5
        ours = lambda q, k, v, mod=mod: mod.run(q, k, v, jnp.zeros_like(q), scale, interpret=small)[0]
        # XLA's attention takes [batch, seq, heads, dim].
        t = lambda x: x.transpose(1, 0, 2)[None]
        xla = lambda q, k, v: jax.nn.dot_product_attention(t(q), t(k), t(v), scale=scale, is_causal=True)[0].transpose(
            1, 0, 2
        )
        cases.append((f"attention S{s}", ours, (q, k, v), xla, (q, k, v), 2 * 2 * h * s * s * d // 2))
    for label, ours, oargs, xla, xargs, flops in cases:
        try:
            got, med, mn = timed(ours, oargs, a.iters)
        except Exception as e:
            print(json.dumps({"case": label, "error": f"{type(e).__name__}: {str(e)[:3000]}"}), flush=True)
            continue
        want, xmed, xmn = timed(xla, xargs, a.iters)
        g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
        err = float(np.abs(g - w).max() / np.abs(w).max())
        print(
            json.dumps(
                {
                    "device": dev.device_kind,
                    "case": label,
                    "tessel_ms": round(med, 4),
                    "tessel_min_ms": round(mn, 4),
                    "xla_ms": round(xmed, 4),
                    "xla_min_ms": round(xmn, 4),
                    "tessel_tflops": round(flops / med / 1e9, 2),
                    "xla_tflops": round(flops / xmed / 1e9, 2),
                    "max_rel_diff": err,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
