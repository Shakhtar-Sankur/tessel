#!/bin/bash
# tessel's LLM engine on a Kaggle or Colab GPU: builds tessel, runs the
# engine's tests on the GPU, then TinyLlama-1.1B-Chat against Hugging Face
# transformers (scripts/llm_bench.py), vLLM (scripts/vllm_bench.py, in its
# own environment) and llama.cpp (built with CUDA; scripts/llamacpp_bench.py).
# Writes llm_report.txt. vLLM's install and llama.cpp's build take a while;
# NO_VLLM=1 or NO_LLAMACPP=1 skips them.
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
python3 scripts/llm_bench.py --json "$OUT" 2>&1 | grep -v "^Warning\|warn(\|^\[transformers\] Both" | tee -a "$REPORT"

step "vLLM on the same prompts (its own environment; skip with NO_VLLM=1)"
if [ -z "${NO_VLLM:-}" ]; then
  (
    V=/tmp/vllm-env
    LOG=bench/results/vllm_install.log
    pip install --quiet uv > $LOG 2>&1
    UV="python3 -m uv"
    { $UV venv --clear $V --python 3.12 && $UV pip install --python $V/bin/python vllm; } >> $LOG 2>&1
    $V/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" 2>> $LOG \
      || { echo "vLLM install failed:"; grep -v '^\s*$' $LOG | tail -n 15; exit 0; }
    for b in 1 8 32; do
      timeout 1200 $V/bin/python scripts/vllm_bench.py --batch "$b" --json "$OUT" 2> bench/results/vllm.err | grep '^{' \
        || { echo "vLLM batch $b failed:"; grep -v '^\s*$' bench/results/vllm.err | tail -n 8; }
    done
  ) 2>&1 | tee -a "$REPORT"
fi

step "llama.cpp, CUDA build, f16 GGUF (skip with NO_LLAMACPP=1)"
if [ -z "${NO_LLAMACPP:-}" ]; then
  (
    export PATH=/usr/local/cuda/bin:$PATH
    command -v nvcc > /dev/null || { echo "llama.cpp skipped: no nvcc (CUDA toolkit) on this machine"; exit 0; }
    L=/tmp/llama.cpp
    [ -d $L ] || git clone --quiet --depth 1 https://github.com/ggml-org/llama.cpp $L
    git -C $L log -1 --format='llama.cpp %h %cs'
    nvcc --version | tail -n 1
    # GGML_CUDA_NO_VMM: no link against the driver library, which some
    # toolkits (Kaggle's) lack; it only changes how llama.cpp pools memory.
    # A build from earlier in the same session is reused.
    if [ ! -x $L/build/bin/llama-batched-bench ]; then
    rm -rf $L/build
    cmake -S $L -B $L/build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > $L/cmake.log 2>&1 \
      || { echo "llama.cpp configure failed:"; tail -n 15 $L/cmake.log; exit 0; }
    echo "compiling llama.cpp's CUDA kernels: 30-60 minutes on a few CPU cores; progress every minute"
    cmake --build $L/build --target llama-bench llama-batched-bench -j"$(nproc)" > $L/build.log 2>&1 &
    BUILD=$!
    while kill -0 $BUILD 2> /dev/null; do
      sleep 60
      echo "  $(date +%H:%M) $(grep -o '^\[ *[0-9]*%\]' $L/build.log | tail -n 1) $(grep -c 'Building CUDA' $L/build.log) CUDA files compiled"
    done
    wait $BUILD || { echo "llama.cpp build failed:"; grep -i -m 10 "error" $L/build.log; tail -n 5 $L/build.log; exit 0; }
    fi
    pip install --quiet $L/gguf-py sentencepiece > /dev/null 2>&1
    MODEL=$(python3 -c "import json; print(json.load(open('bench/results/llm_prompts.json'))['path'])")
    python3 $L/convert_hf_to_gguf.py "$MODEL" --outtype f16 --outfile /tmp/model-f16.gguf > $L/convert.log 2>&1 \
      || { echo "GGUF conversion failed:"; tail -n 10 $L/convert.log; exit 0; }
    python3 scripts/llamacpp_bench.py --bin $L/build/bin --gguf /tmp/model-f16.gguf --json "$OUT" \
      || echo "llama.cpp benchmark failed"
  ) 2>&1 | tee -a "$REPORT"
fi

step "raw results (gzip + base64 of $OUT)"
python3 -c "import gzip,base64;print(base64.b64encode(gzip.compress(open('$OUT','rb').read(),9)).decode())" | tee -a "$REPORT"
log
log "TESSEL LLM REPORT END"
