#!/bin/bash
# tessel on a Google Colab GPU (or any Linux machine with an NVIDIA GPU and
# Python with torch): builds tessel, runs the test suite on the GPU, then
# benchmarks tessel's kernels against cuBLAS, PyTorch and Triton, and writes
# everything to colab_report.txt (printed at the end).
#
# In Colab (Runtime > Change runtime type > T4 GPU), one cell:
#   !git clone https://github.com/Shakhtar-Sankur/tessel && cd tessel && bash scripts/colab.sh
# Again in the same session:
#   !cd tessel && git pull && bash scripts/colab.sh
#
# Usage: bash scripts/colab.sh [ROUNDS]
set -u
ROUNDS=${1:-2}
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

step "benchmark ($ROUNDS rounds)"
OUT=bench/results/gpu_runs.jsonl
mkdir -p bench/results
: > "$OUT"
for r in $(seq "$ROUNDS"); do
  ./target/release/tessel bench --json "$OUT" > /dev/null || log "tessel bench failed"
  python3 scripts/bench_baselines.py --json "$OUT" > /dev/null 2> bench/results/baselines.err || log "baselines failed: $(tail -1 bench/results/baselines.err)"
  log "round $r done"
done
grep -h '"error"' "$OUT" | head -5 | tee -a "$REPORT"

step "summary"
python3 scripts/summarize.py "$OUT" 2>&1 | tee -a "$REPORT"

step "raw results (gzip + base64 of $OUT)"
python3 -c "import gzip,base64;print(base64.b64encode(gzip.compress(open('$OUT','rb').read(),9)).decode())" | tee -a "$REPORT"
log
log "TESSEL COLAB REPORT END"
