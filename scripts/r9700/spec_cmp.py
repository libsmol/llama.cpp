#!/usr/bin/env python3
# spec_cmp.py ref tag...: greedy token agreement against spec-<ref>.json
import json, sys
L = "/home/hxcliff/src/inkling-work/logs/spec-%s.json"
a = json.load(open(L % sys.argv[1]))
for tag in sys.argv[2:]:
    b = json.load(open(L % tag))
    out = []
    for k in a:
        x, y = a[k]["tokens"], b[k]["tokens"]
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
        out.append(f"{k}: " + ("identical" if n is None and len(x) == len(y) else f"diverge@{n if n is not None else min(len(x), len(y))}"))
    print(f"{tag:10s} vs {sys.argv[1]}: " + ", ".join(out))
