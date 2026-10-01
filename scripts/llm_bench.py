"""tessel's LLM engine against Hugging Face transformers on one GPU.

Downloads TinyLlama-1.1B-Chat (or --model), builds chat prompts, and:

1. Correctness: the logits after the first prompt, tessel's against
   transformers' in float32; then greedy generations of both (fp16 for
   transformers), compared token by token.
2. Throughput: greedy generation of every prompt (--max-new tokens each),
   by tessel (continuous batching, at each --batch) and by transformers
   (generate on a left-padded batch of the same size), as generated tokens
   per second; and one sequence alone, as milliseconds per token.

Prints one JSON line per measurement; with --json, appends them to a file.
usage: python scripts/llm_bench.py [--tessel target/release/tessel] [--json OUT]
"""

import argparse
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import torch

QUESTIONS = [
    "Explain how a hash map handles collisions.",
    "Write a haiku about the ocean at night.",
    "What causes the seasons on Earth?",
    "Give three tips for writing clear technical documentation.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "How does a transistor work?",
    "What is the difference between TCP and UDP?",
    "Describe the water cycle to a ten-year-old.",
    "Why is the sky blue?",
    "List five uses of a paperclip.",
    "What is gradient descent?",
    "Write a short story opening about a lighthouse keeper.",
    "How do vaccines train the immune system?",
    "Compare Python lists and tuples.",
    "What makes a good password?",
    "Explain recursion with an example.",
    "What is the capital of Australia, and why was it chosen?",
    "How do airplanes stay in the air?",
    "Give a recipe for a simple tomato soup.",
    "What is a black hole?",
    "Explain what a database index is.",
    "Write a limerick about a cat who codes.",
    "How does compound interest work?",
    "What are the main causes of inflation?",
    "Describe how photosynthesis works.",
    "What is the difference between a virus and a bacterium?",
    "Explain the Pythagorean theorem.",
    "How does a refrigerator keep food cold?",
    "What is version control and why use it?",
    "Give advice for a first job interview.",
    "How do noise-cancelling headphones work?",
    "What is the role of mitochondria in a cell?",
]


def log(out, row):
    line = json.dumps(row)
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def tessel(exe, model, prompts, max_new, batch, extra=()):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prompts.json")
        o = os.path.join(d, "out.json")
        json.dump(prompts, open(p, "w"))
        cmd = [exe, "llm", model, "--prompts", p, "--max-new", str(max_new), "--batch", str(batch),
               "--warmup", "--json", o, *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"tessel llm failed: {r.stderr.strip()[-2000:]}")
        return json.loads(open(o).read())


def load_model(path, dtype):
    """The model on the GPU; newer transformers call the argument dtype."""
    from transformers import AutoModelForCausalLM

    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return m.to(dtype).cuda().eval()


def hf_generate(model, tok, prompts, max_new, batch, eos):
    """Greedy generation in left-padded batches; (outputs, seconds)."""
    outs = []
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(0, len(prompts), batch):
        group = prompts[i : i + batch]
        n = max(len(p) for p in group)
        ids = torch.tensor([[tok.pad_token_id] * (n - len(p)) + p for p in group], device="cuda")
        mask = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in group], device="cuda")
        g = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=eos)
        for row in g[:, n:].tolist():
            # Up to and including the first end-of-sequence token, as tessel stops.
            out = []
            for x in row:
                out.append(x)
                if x == eos:
                    break
            outs.append(out)
    torch.cuda.synchronize()
    return outs, time.perf_counter() - t


def divergences(model, prompts, xs, ys):
    """For each pair of generations that differ: where, the two tokens, and
    transformers' (fp16) top-two logit margin and the gap between the two
    tokens' logits there, given the shared history."""
    out = []
    for p, x, y in zip(prompts, xs, ys):
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]:
            k += 1
        if x == y or k >= min(len(x), len(y)):
            continue
        with torch.no_grad():
            lg = model(torch.tensor([p + x[:k]], device="cuda")).logits[0, -1].float()
        top = torch.topk(lg, 2).values.tolist()
        out.append({"at": k, "a": x[k], "b": y[k], "top2_margin": top[0] - top[1],
                    "logit_gap": abs(float(lg[x[k]] - lg[y[k]])), "logit_scale": float(lg.abs().max())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tessel", default="target/release/tessel")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--json")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(a.model, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # The chat template as text, then its tokens: plain lists of ints on any
    # transformers version (newer ones return a BatchEncoding when asked to
    # tokenize). The template holds its own special tokens.
    prompts = []
    for q in QUESTIONS:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        prompts.append([int(x) for x in tok(text, add_special_tokens=False)["input_ids"]])
    eos = tok.eos_token_id
    dev = torch.cuda.get_device_name(0)
    # The prompts, for the other engines' scripts (vLLM, llama.cpp).
    if a.json:
        with open(os.path.join(os.path.dirname(a.json) or ".", "llm_prompts.json"), "w") as f:
            json.dump({"model": a.model, "path": path, "eos": eos, "max_new": a.max_new, "prompts": prompts}, f)
    base = {"device": dev, "model": a.model}

    # 1. Logits after the first prompt: tessel against transformers in fp32.
    t_out = tessel(a.tessel, path, prompts[:1], 1, 1,
                   ["--logits", os.path.join(tempfile.gettempdir(), "tessel_logits.json")])
    ours = np.array(json.load(open(os.path.join(tempfile.gettempdir(), "tessel_logits.json"))), np.float64)
    m32 = load_model(path, torch.float32)
    with torch.no_grad():
        ref = m32(torch.tensor([prompts[0]], device="cuda")).logits[0, -1].double().cpu().numpy()
    del m32
    torch.cuda.empty_cache()
    top5 = len(set(np.argsort(-ours)[:5]) & set(np.argsort(-ref)[:5]))
    log(a.json, {**base, "kind": "logits", "max_abs_diff": float(np.abs(ours - ref).max()),
                 "rel_to_max": float(np.abs(ours - ref).max() / np.abs(ref).max()),
                 "top1_same": bool(ours.argmax() == ref.argmax()), "top5_overlap": top5})

    # 2. Throughput and generations.
    model = load_model(path, torch.float16)
    for b in [int(x) for x in a.batches.split(",")]:
        ps = prompts[:4] if b == 1 else prompts
        t = tessel(a.tessel, path, ps, a.max_new, b)
        log(a.json, {**base, "kind": "generate", "engine": "tessel", "batch": b, "requests": len(ps),
                     "generated": t["generated"], "seconds": t["seconds"], "tokens_per_s": t["tokens_per_s"],
                     "decode_tokens_per_s": t["decode_tokens_per_s"], "warmup_s": t["warmup_s"]})
        if b > 1:
            # With each prompt in a prefill step of its own: what packing saves.
            npk = tessel(a.tessel, path, ps, a.max_new, b, ["--no-pack"])
            log(a.json, {**base, "kind": "generate", "engine": "tessel-no-pack", "batch": b, "requests": len(ps),
                         "generated": npk["generated"], "seconds": npk["seconds"], "tokens_per_s": npk["tokens_per_s"],
                         "decode_tokens_per_s": npk["decode_tokens_per_s"],
                         # Packing must change only the speed, never a token.
                         "identical_to_packed": sum(x == y for x, y in zip(npk["outputs"], t["outputs"])),
                         # Where it does change one, how close the call was:
                         # the margin between transformers' top two logits at
                         # the first differing token (a near-tie is rounding).
                         "packing_divergences": divergences(model, ps, t["outputs"], npk["outputs"])})
        if b in (1, 8, 32):
            # Where tessel's decode time goes, kernel by kernel.
            pr = tessel(a.tessel, path, ps, a.max_new, b, ["--profile"])
            dec = [r for r in pr.get("profile", []) if r["phase"] == "decode"]
            tot = sum(r["ms"] for r in dec) or 1.0
            log(a.json, {**base, "kind": "profile", "engine": "tessel", "batch": b, "decode_ms": tot,
                         "kernels": [{**r, "pct": round(100 * r["ms"] / tot, 1)} for r in dec]})
        hf_generate(model, tok, ps[:b], 8, b, eos)  # warm up
        outs, secs = hf_generate(model, tok, ps, a.max_new, b, eos)
        n = sum(len(o) for o in outs)
        log(a.json, {**base, "kind": "generate", "engine": "transformers", "batch": b, "requests": len(ps),
                     "generated": n, "seconds": secs, "tokens_per_s": n / secs})
        if b == 1:
            # The same without CUDA graphs: what replaying each decode step's
            # launches as one graph saves.
            ng = tessel(a.tessel, path, ps, a.max_new, b, ["--no-graphs"])
            log(a.json, {**base, "kind": "generate", "engine": "tessel-no-graphs", "batch": b, "requests": len(ps),
                         "generated": ng["generated"], "seconds": ng["seconds"], "tokens_per_s": ng["tokens_per_s"],
                         "decode_tokens_per_s": ng["decode_tokens_per_s"]})
            # And with the default matmul tiles: what tuning them on the GPU gains.
            nt = tessel(a.tessel, path, ps, a.max_new, b, ["--no-tune"])
            log(a.json, {**base, "kind": "generate", "engine": "tessel-no-tune", "batch": b, "requests": len(ps),
                         "generated": nt["generated"], "seconds": nt["seconds"], "tokens_per_s": nt["tokens_per_s"],
                         "decode_tokens_per_s": nt["decode_tokens_per_s"]})
            if a.json:
                with open(os.path.join(os.path.dirname(a.json) or ".", "llm_tessel_outputs.json"), "w") as f:
                    json.dump(t["outputs"], f)
            # Token-by-token agreement of the greedy generations (fp16 both).
            same = []
            for x, y in zip(t["outputs"], outs):
                k = 0
                while k < min(len(x), len(y)) and x[k] == y[k]:
                    k += 1
                same.append(k)
            # Where a generation differs: the margin between transformers' top
            # two logits (fp16) at that token, given the shared history. A
            # small margin is a near-tie that rounding can tip either way.
            ties = []
            for p, x, y, k in zip(ps, t["outputs"], outs, same):
                if x != y and k < min(len(x), len(y)):
                    with torch.no_grad():
                        lg = model(torch.tensor([p + y[:k]], device="cuda")).logits[0, -1].float()
                    top = torch.topk(lg, 2).values.tolist()
                    ties.append({"at": k, "tessel": x[k], "transformers": y[k], "top2_margin": top[0] - top[1],
                                 "logit_gap_between_choices": float(lg[y[k]] - lg[x[k]])})
            log(a.json, {**base, "kind": "agreement", "prompts": len(same), "max_new": a.max_new, "divergences": ties,
                         "matching_prefix": same, "identical": sum(x == y for x, y in zip(t["outputs"], outs)),
                         "sample_tessel": tok.decode(t["outputs"][0]), "sample_transformers": tok.decode(outs[0])})


if __name__ == "__main__":
    main()
