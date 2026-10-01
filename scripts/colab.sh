#!/bin/bash
# tessel on a Google Colab GPU (or any Linux machine with an NVIDIA GPU and
# Python with torch): builds tessel, runs the test suite on the GPU, then
# benchmarks tessel's kernels against cuBLAS, PyTorch and Triton, and writes
# everything to colab_report.txt (printed at the end).
#
# On Kaggle, the same cell works in a notebook with Accelerator "GPU T4 x2"
# and Internet on (it uses the first GPU).
#
# In Colab (Runtime > Change runtime type > T4 GPU), one cell:
#   !git clone https://github.com/Shakhtar-Sankur/tessel && cd tessel && bash scripts/colab.sh
# Again in the same session:
#   !cd tessel && git pull && bash scripts/colab.sh
#
# Usage: bash scripts/colab.sh [ROUNDS]
set -u
ROUNDS=${1:-4}
cd "$(dirname "$0")/.."
REPORT=$PWD/colab_report.txt
: > "$REPORT"
log() { echo "$@" | tee -a "$REPORT"; }
step() { log; log "=== $* ==="; }

step "machine"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv 2>&1 | tee -a "$REPORT"
git log -1 --format='tessel %h %s' | tee -a "$REPORT"

step "toolchain"
if ! command -v cargo > /dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1
fi
source "$HOME/.cargo/env"
rustc --version | tee -a "$REPORT"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tee -a "$REPORT"
python3 -c "import triton; print('triton', triton.__version__)" 2>&1 | tail -1 | tee -a "$REPORT"

step "build"
cargo build --release 2>&1 | tail -2 | tee -a "$REPORT"

step "tests (every kernel on the emulator and on this GPU)"
cargo test --release 2>&1 | grep -E "^test |test result|panicked|error" | tee -a "$REPORT"

step "benchmark ($ROUNDS rounds; tessel tunes in the first, then reuses its choices)"
# The T4 throttles as it heats (70 W cap), so which side runs first in a
# round matters: the order alternates between rounds, every row records
# its round, and the GPU's clocks are sampled every 200 ms while each side
# runs (summarized per side in the report).
OUT=bench/results/gpu_runs.jsonl
TUNED=bench/results/tuned.txt
TMP=bench/results/phase.jsonl
mkdir -p bench/results
: > "$OUT"
: > "$TUNED"
phase() { # round, side, command...
  local r=$1 side=$2
  shift 2
  : > "$TMP"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 200 > bench/results/clocks.csv 2> /dev/null &
  local smi=$!
  "$@" > /dev/null 2> bench/results/phase.err || echo "round $r: $side failed: $(tail -1 bench/results/phase.err)"
  kill $smi 2> /dev/null
  wait $smi 2> /dev/null
  python3 - "$r" "$side" "$TMP" bench/results/clocks.csv "$OUT" <<'PY'
import json, statistics, sys
r, side, tmp, clocks, out = sys.argv[1:]
c = [l.split(",") for l in open(clocks) if l.count(",") == 2]
c = [[float(x) for x in l] for l in c if all(x.strip().replace(".", "").isdigit() for x in l)]
clock = {"round": int(r), "kind": "clocks", "engine": side, "samples": len(c)}
if c:
    clock.update(sm_mhz_median=statistics.median(x[0] for x in c), sm_mhz_min=min(x[0] for x in c),
                 watts_median=statistics.median(x[1] for x in c), temp_c_max=max(x[2] for x in c))
with open(out, "a") as f:
    for l in open(tmp):
        if l.strip():
            row = json.loads(l)
            row["round"] = int(r)
            f.write(json.dumps(row) + "\n")
    f.write(json.dumps(clock) + "\n")
print(f"round {r}: {side}: SM clock median {clock.get('sm_mhz_median', '?')} MHz (min {clock.get('sm_mhz_min', '?')}), "
      f"{clock.get('watts_median', '?')} W, up to {clock.get('temp_c_max', '?')} C, {clock['samples']} samples")
PY
}
for r in $(seq "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then
    phase "$r" tessel ./target/release/tessel bench --json "$TMP" --tuned "$TUNED" | tee -a "$REPORT"
    phase "$r" baselines python3 scripts/bench_baselines.py --json "$TMP" | tee -a "$REPORT"
  else
    phase "$r" baselines python3 scripts/bench_baselines.py --json "$TMP" | tee -a "$REPORT"
    phase "$r" tessel ./target/release/tessel bench --json "$TMP" --tuned "$TUNED" | tee -a "$REPORT"
  fi
done
python3 -c "
import json, sys
for l in open('$OUT'):
    r = json.loads(l)
    if 'error' in r:
        print(l.strip())
" | head -5 | tee -a "$REPORT"

step "summary"
python3 scripts/summarize.py "$OUT" 2>&1 | tee -a "$REPORT"

step "raw results (gzip + base64 of $OUT)"
python3 -c "import gzip,base64;print(base64.b64encode(gzip.compress(open('$OUT','rb').read(),9)).decode())" | tee -a "$REPORT"
log
log "TESSEL COLAB REPORT END"
