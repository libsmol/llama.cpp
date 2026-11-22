#!/usr/bin/env python3
import sys, json, numpy as np
sys.path.insert(0, "/home/hxcliff/src/llama.cpp-r9700/gguf-py")
from gguf import GGUFReader
a = GGUFReader(sys.argv[1]); b = GGUFReader(sys.argv[2])
ta = {t.name: t for t in a.tensors}; tb = {t.name: t for t in b.tensors}
inv = {"q": "attn", "k": "k", "v": "v", "ffn": "mlp"}
res = np.asarray(ta["loop_res_query.weight"].data).reshape(-1, 2304); kv = np.asarray(ta["loop_kv_res_query.weight"].data).reshape(-1, 2304)
bad = 0; n = 0
for name, t in tb.items():
    p = name.split(".")
    if p[0] == "blk" and p[2] == "loop_msg_query":
        i = int(p[1]) - 6
        exp = np.stack([res[2*i], kv[2*i], kv[2*i + 1], res[2*i + 1]])
        ok = np.array_equal(np.asarray(t.data).reshape(4, 2304), exp)
    else:
        src = name
        if p[0] == "blk" and p[2].startswith("loop_") and "_ctrl_" in p[2]:
            mod, kind = p[2][5:].split("_ctrl_")
            src = f"blk.{p[1]}.loop_{inv[mod]}_{kind}.weight"
        s = ta[src]
        ok = s.tensor_type == t.tensor_type and list(s.shape) == list(t.shape) and np.array_equal(np.asarray(s.data).view(np.uint8), np.asarray(t.data).view(np.uint8))
    n += 1; bad += 0 if ok else 1
    if not ok: print("MISMATCH", name)
off = set(json.load(open(sys.argv[3])))
print(f"{n} tensors checked, {bad} mismatches; missing vs official: {sorted(off - set(tb))[:5]}; extra vs official: {len(set(tb) - off)} (expected 40 .scale)")
print("extra:", sorted(set(tb) - off)[:3], "...")
