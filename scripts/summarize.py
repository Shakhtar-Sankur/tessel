"""Tables from the JSON lines of `tessel bench` and bench_baselines.py.

Every (case, engine) can appear several times (one per round); its median
time is the median of the rounds' medians. Usage: python scripts/summarize.py RESULTS.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict

KINDS = [
    ("gemm", "GEMM, fp16 in and out, fp32 accumulation (M = N = K)", "tflops", "TFLOPS"),
    ("attention", "Causal attention, fp16, 32 heads, head dimension 64", "tflops", "TFLOPS"),
    ("softmax", "Softmax over rows, fp32", "gbps", "GB/s"),
    ("rmsnorm", "RMSNorm over rows, fp16", "gbps", "GB/s"),
]
ORDER = ["tessel", "cublas", "torch-sdpa", "triton", "torch.compile", "torch"]
NAMES = {"tessel": "tessel", "cublas": "cuBLAS (torch.matmul)", "torch-sdpa": "PyTorch SDPA", "triton": "Triton",
         "torch.compile": "torch.compile", "torch": "PyTorch"}


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "median_ms" in r]
    dev = rows[0]["device"] if rows else "?"
    print(f"GPU: {dev}\n")
    for kind, title, metric, unit in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        labels = list(dict.fromkeys(r["label"] for r in rs))
        engines = [e for e in ORDER if any(r["engine"] == e for r in rs)]
        cell = defaultdict(list)
        for r in rs:
            cell[(r["label"], r["engine"])].append(r)
        print(f"{title}: median ms ({unit})\n")
        print("| Size | " + " | ".join(NAMES[e] for e in engines) + " |")
        print("|---" * (len(engines) + 1) + "|")
        for lab in labels:
            vals = {}
            for e in engines:
                c = cell.get((lab, e))
                if c:
                    med = statistics.median(x["median_ms"] for x in c)
                    rate = statistics.median(x[metric] for x in c)
                    vals[e] = (med, rate)
            best = min(v[0] for v in vals.values())
            out = []
            for e in engines:
                if e not in vals:
                    out.append("—")
                    continue
                med, rate = vals[e]
                s = f"{med:.3f} ({rate:.1f})" if metric == "tflops" else f"{med:.3f} ({rate:.0f})"
                out.append(f"**{s}**" if med == best else s)
            print(f"| {lab} | " + " | ".join(out) + " |")
        errs = defaultdict(float)
        for r in rs:
            errs[r["engine"]] = max(errs[r["engine"]], r.get("rel_err", 0.0))
        print("\nLargest error against a float64 reference (relative to the largest value): "
              + ", ".join(f"{NAMES[e]} {errs[e]:.1e}" for e in engines) + "\n")
    tuned = [r for r in rows if r["engine"] == "tessel"]
    if tuned:
        print("tessel's tuned configurations:")
        for r in tuned:
            print(f"- {r['kind']} {r['label']}: {r['config']}")


if __name__ == "__main__":
    main(sys.argv[1])
