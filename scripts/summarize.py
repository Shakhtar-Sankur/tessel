"""Tables from the JSON lines of `tessel bench` and bench_baselines.py.

Every (case, engine) can appear several times (one per round); its median
time is the median of the rounds' medians. Where rows record their round,
tessel is also compared round by round with the fastest other engine of
the same round (the GPU's clocks drift between rounds as it heats), and
the clocks sampled while each side ran are summarized.
Usage: python scripts/summarize.py RESULTS.jsonl
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
        paired(rs, labels)
    notes = [json.loads(l) for l in open(path) if l.strip() and '"note"' in l]
    for n in notes[-1:]:
        print(f"Triton {n['triton_version']}: its matmul PTX ({n['matmul_ptx_files']} files) "
              + ("uses" if n["matmul_uses_mma"] else "does NOT use") + " tensor-core mma instructions on this GPU.\n")
    clocks = [json.loads(l) for l in open(path) if l.strip() and '"clocks"' in l]
    clocks = [c for c in clocks if c.get("samples")]
    if clocks:
        print("GPU clocks while each side ran (median of the rounds' medians; 200 ms samples):")
        for side in ("tessel", "baselines"):
            cs = [c for c in clocks if c["engine"] == side]
            if cs:
                print(f"- {side}: SM {statistics.median(c['sm_mhz_median'] for c in cs):.0f} MHz "
                      f"(lowest sample {min(c['sm_mhz_min'] for c in cs):.0f}), "
                      f"{statistics.median(c['watts_median'] for c in cs):.1f} W, up to {max(c['temp_c_max'] for c in cs):.0f} C")
        print()
    tuned = {}
    for r in rows:
        if r["engine"] == "tessel":
            tuned[(r["kind"], r["label"])] = r["config"]
    if tuned:
        print("tessel's tuned configurations:")
        for (kind, label), config in tuned.items():
            print(f"- {kind} {label}: {config}")


def paired(rs, labels):
    """tessel against the fastest other engine, round by round."""
    if not all("round" in r for r in rs):
        return
    lines = []
    for lab in labels:
        ratios = []
        for rnd in sorted({r["round"] for r in rs}):
            here = [r for r in rs if r["label"] == lab and r["round"] == rnd]
            ours = [r["median_ms"] for r in here if r["engine"] == "tessel"]
            others = [(r["median_ms"], r["engine"]) for r in here if r["engine"] != "tessel"]
            if ours and others:
                best = min(others)
                ratios.append((best[0] / ours[0], best[1]))
        if ratios:
            xs = [x for x, _ in ratios]
            who = statistics.mode(e for _, e in ratios)
            lines.append(f"- {lab}: {statistics.median(xs):.2f}x (rounds: {', '.join(f'{x:.2f}' for x in xs)}) "
                         f"against {NAMES[who]}")
    if lines:
        print("tessel's speed relative to the fastest other engine in the same round (above 1 is faster):")
        print("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
