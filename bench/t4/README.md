# Raw results on a Colab Tesla T4

One JSON line per (case, engine, round), exactly as `scripts/colab.sh`
wrote them (`bench/results/gpu_runs.jsonl`), decoded from the report's
base64. `python3 scripts/summarize.py FILE` turns a file into the tables.

| File | tessel commit | Rounds | Notes |
|---|---|---|---|
| run1_fa918eb.jsonl | fa918eb | 2 | tessel ran before the baselines in every round |
| run2_3966b0e.jsonl | 3966b0e | 3 | as run 1; in round 1 tessel also tuned, on a GPU still at idle clocks |

These runs predate round tags in the rows: in both, each round wrote the
tessel rows and then the baseline rows, in order. From the next run on,
rows carry their round, the order alternates between rounds, and the
GPU's clocks are recorded while each side runs.
