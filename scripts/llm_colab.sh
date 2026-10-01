#!/bin/bash
# tessel's LLM engine on a Kaggle or Colab GPU: builds tessel, runs the
# engine's tests on the GPU, then TinyLlama-1.1B-Chat against Hugging Face
# transformers (scripts/llm_bench.py). Writes llm_report.txt.
#
# Kaggle (Accelerator "GPU T4 x2", Internet on) or Colab (T4 GPU), one cell:
#   !git clone https://github.com/Shakhtar-Sankur/tessel && cd tessel && bash scripts/llm_colab.sh
set -u
cd "$(dirname "$0")/.."
REPORT=$PWD/llm_report.txt
: > "$REPORT"
log() { echo "$@" | tee -a "$REPORT"; }
step() { log; log "=== $* ==="; }

step "machine"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | tee -a "$REPORT"
git log -1 --format='tessel %h %s' | tee -a "$REPORT"
python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" 2>&1 | tee -a "$REPORT"

step "build"
if ! command -v cargo > /dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1
fi
source "$HOME/.cargo/env"
cargo build --release 2>&1 | tail -1 | tee -a "$REPORT"

step "engine tests"
cargo test --release --test llm 2>&1 | grep -E "^test |test result|panicked" | tee -a "$REPORT"

step "TinyLlama-1.1B: tessel against transformers"
OUT=bench/results/llm_runs.jsonl
mkdir -p bench/results
: > "$OUT"
python3 scripts/llm_bench.py --json "$OUT" 2>&1 | grep -v "^Warning\|warn(" | tee -a "$REPORT"

step "raw results (gzip + base64 of $OUT)"
python3 -c "import gzip,base64;print(base64.b64encode(gzip.compress(open('$OUT','rb').read(),9)).decode())" | tee -a "$REPORT"
log
log "TESSEL LLM REPORT END"
