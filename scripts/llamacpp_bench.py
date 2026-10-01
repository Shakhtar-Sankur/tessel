"""llama.cpp (CUDA build, f16 GGUF of the same model) with its own benchmark
tools, for comparison with tessel's decode speed:

- llama-bench: tokens per second generating 128 tokens for one sequence
  (tg128) and processing a 64-token prompt (pp64);
- llama-batched-bench: decode tokens per second (S_TG) with 1, 8 and 32
  sequences, each with a 64-token prompt and 128 new tokens.

Appends one JSON line per measurement to --json.
usage: python scripts/llamacpp_bench.py --bin /tmp/llama.cpp/build/bin --gguf model.gguf --json OUT
"""

import argparse
import json
import subprocess


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()[-1500:]}")
    return r.stdout


def table(text):
    """The rows of llama-batched-bench's markdown table, as dicts."""
    rows, head = [], None
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        if head is None and "S_TG t/s" in cells:
            head = cells
        elif head and len(cells) == len(head) and cells[0].replace(".", "").isdigit():
            rows.append(dict(zip(head, cells)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--device", default="Tesla T4")
    a = ap.parse_args()
    out = open(a.json, "a")

    def log(row):
        row = {"device": a.device, "engine": "llama.cpp", **row}
        line = json.dumps(row)
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()

    rs = json.loads(run([f"{a.bin}/llama-bench", "-m", a.gguf, "-ngl", "99", "-p", "64", "-n", "128", "-r", "5", "-o", "json"]))
    for r in rs:
        log({"kind": "llama-bench", "test": f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}",
             "tokens_per_s": r["avg_ts"], "stddev": r["stddev_ts"], "build": r.get("build_commit", "")})
    text = run([f"{a.bin}/llama-batched-bench", "-m", a.gguf, "-ngl", "99", "-c", "16384",
                "-npp", "64", "-ntg", "128", "-npl", "1,8,32"])
    for r in table(text):
        log({"kind": "batched-bench", "batch": int(r["B"]), "prompt": int(r["PP"]), "gen": int(r["TG"]),
             "decode_tokens_per_s": float(r["S_TG t/s"]), "total_tokens_per_s": float(r["S t/s"])})


if __name__ == "__main__":
    main()
