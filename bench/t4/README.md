# Raw results on a Colab Tesla T4

One JSON line per (case, engine, round), exactly as `scripts/colab.sh`
wrote them (`bench/results/gpu_runs.jsonl`), decoded from the report's
base64. `python3 scripts/summarize.py FILE` turns a file into the tables.

| File | tessel commit | Rounds | Notes |
|---|---|---|---|
| run1_fa918eb.jsonl | fa918eb | 2 | tessel ran before the baselines in every round |
| run2_3966b0e.jsonl | 3966b0e | 3 | as run 1; in round 1 tessel also tuned, on a GPU still at idle clocks |
| llm_run1_2fd72a1.jsonl | 2fd72a1 | 1 | Kaggle T4; the LLM engine on TinyLlama-1.1B against transformers (scripts/llm_bench.py) |
| run3_c7ab694.jsonl | c7ab694 | 4 | Kaggle T4; swizzled staging; order alternating; rows carry their round, and clock samples per side |

Runs 1 and 2 predate round tags in the rows: in both, each round wrote
the tessel rows and then the baseline rows, in order. From run 3 on, rows
carry their round and the order alternates between rounds. Run 3's
"clocks" rows sample the GPU every 200 ms over each side's whole run,
which is mostly idle time between timed loops (compiling, checking
results), so they read far below the clock the timed launches ran at;
later runs also record, in every row, the SM clock read right after its
timed launches.
