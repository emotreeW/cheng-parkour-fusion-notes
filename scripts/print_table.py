#!/usr/bin/env python
"""Print mxd_mean by tag and terrain from a jsonl eval log."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/cheng_eval.jsonl"
rows = defaultdict(dict)
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tag = r.get("tag") or r.get("exptid")
        key = (tag, r.get("fusion_mode"), r.get("checkpoint"))
        rows[key][r["terrain"]] = r.get("mxd_mean")
print("tag fusion ckpt | mixed hurdle step gap ramps")
for key, d in rows.items():
    tag, mode, ckpt = key
    vals = " ".join("{:.3f}".format(d[t]) if t in d else "n/a" for t in ("mixed", "hurdle", "step", "gap", "ramps"))
    print(tag, mode, ckpt, "|", vals)
