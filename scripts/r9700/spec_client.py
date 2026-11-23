#!/usr/bin/env python3
# spec_client.py <tag> [port]: greedy completions through the chat template; timings, draft acceptance, tokens -> spec-<tag>.json
import json, sys, urllib.request
tag = sys.argv[1]; port = sys.argv[2] if len(sys.argv) > 2 else "18080"
base = f"http://127.0.0.1:{port}"
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))
import os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/llama-context.cpp")).read()[:60000]
cases = [
    ("code",  "Write a Python function that parses an ISO 8601 duration string such as P3DT4H5M6S into total seconds, with input validation and a few doctests."),
    ("chat",  "用通俗的语言解释一下为什么天空是蓝色的，再说说为什么日落时是红色的。"),
    ("json",  "Produce a JSON array of 12 objects describing the planets and dwarf planets of the solar system, with fields name, type, mean_radius_km, moons, and a one-sentence note."),
    ("long",  "Here is a C++ source file:\n\n" + src + "\n\nSummarize what the function llama_context::decode does, step by step."),
]
res = {}
for k, msg in cases:
    prompt = post("/apply-template", {"messages": [{"role": "user", "content": msg}]})["prompt"]
    d = post("/completion", {"prompt": prompt, "n_predict": 256, "temperature": 0.0, "top_k": 1, "seed": 1,
                              "cache_prompt": False, "return_tokens": True})
    t = d["timings"]
    res[k] = {"tokens": d["tokens"], "timings": t}
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    acc = f"{da}/{dn} ({100.0*da/dn:.0f}%)" if dn else "-"
    print(f"{tag:10s} {k:5s} prompt {t['prompt_n']:6d} @ {t['prompt_per_second']:7.1f} t/s | gen {t['predicted_n']:3d} @ {t['predicted_per_second']:6.1f} t/s | draft {acc}", flush=True)
json.dump(res, open(f"/home/hxcliff/src/inkling-work/logs/spec-{tag}.json", "w"))
