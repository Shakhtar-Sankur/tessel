#!/bin/bash
# tessel on a Google Colab TPU: builds tessel, runs every kernel's Pallas
# version on the TPU against NumPy references, then times them against
# XLA. Writes everything to colab_tpu_report.txt (printed at the end).
#
# In Colab (Runtime > Change runtime type > v5e-1 TPU), one cell:
#   !git clone https://github.com/Shakhtar-Sankur/tessel && cd tessel && bash scripts/colab_tpu.sh
set -u
cd "$(dirname "$0")/.."
REPORT=$PWD/colab_tpu_report.txt
: > "$REPORT"
log() { echo "$@" | tee -a "$REPORT"; }
step() { log; log "=== $* ==="; }

step "machine"
python3 -c "import jax; print('jax', jax.__version__); print(jax.devices())" 2>&1 | tee -a "$REPORT"
git log -1 --format='tessel %h %s' | tee -a "$REPORT"

step "toolchain"
if ! command -v cargo > /dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1
fi
source "$HOME/.cargo/env"
rustc --version | tee -a "$REPORT"
cargo build --release 2>&1 | tail -2 | tee -a "$REPORT"

step "correctness on the TPU"
python3 scripts/pallas_check.py --tpu 2>&1 | tee -a "$REPORT"

step "benchmark against XLA"
(cd scripts && python3 pallas_bench.py --tessel ../target/release/tessel) 2>&1 | tee -a "$REPORT"

echo
echo "report: $REPORT"
