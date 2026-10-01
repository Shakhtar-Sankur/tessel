"""vLLM on the same prompts as scripts/llm_bench.py: greedy generation in
fp16, at most --batch sequences at a time (vLLM's max_num_seqs), with its
CUDA graphs on. Prints one JSON line (and appends it to --json): generated
tokens per second over the whole run, as llm_bench measures tessel, and
how many outputs match tessel's batch-1 outputs token for token.

Run it in vLLM's own environment (scripts/llm_colab.sh makes one), after
llm_bench.py has written llm_prompts.json beside --json.
usage: python scripts/vllm_bench.py --batch 8 --json bench/results/llm_runs.jsonl
"""

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    d = os.path.dirname(a.json) or "."
    cfg = json.load(open(os.path.join(d, "llm_prompts.json")))
    prompts = cfg["prompts"][:4] if a.batch == 1 else cfg["prompts"]

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=cfg["path"], dtype="float16", max_num_seqs=a.batch, max_model_len=1024,
              gpu_memory_utilization=0.85, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_new"])
    inputs = [{"prompt_token_ids": p} for p in prompts]
    llm.generate(inputs[: a.batch], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)  # warm up
    t = time.perf_counter()
    outs = llm.generate(inputs, sp, use_tqdm=False)
    secs = time.perf_counter() - t
    ids = [list(o.outputs[0].token_ids) for o in outs]
    n = sum(len(x) for x in ids)
    row = {"device": torch.cuda.get_device_name(0), "model": cfg["model"], "kind": "generate", "engine": "vllm",
           "version": vllm.__version__, "batch": a.batch, "requests": len(prompts), "generated": n,
           "seconds": secs, "tokens_per_s": n / secs}
    mine = os.path.join(d, "llm_tessel_outputs.json")
    if a.batch == 1 and os.path.exists(mine):
        ours = json.load(open(mine))
        row["identical_to_tessel"] = sum(x == y for x, y in zip(ids, ours))
    line = json.dumps(row)
    print(line, flush=True)
    with open(a.json, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
