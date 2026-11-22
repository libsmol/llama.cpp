#!/usr/bin/env python3
# merge_inkling2_attn.py <src.gguf> <dst.gguf>
# per layer: blk.N.attn_qgrkv.weight = rows of attn_q, attn_gate, attn_rel, attn_k, attn_v (same type, same n_embd),
#            blk.N.attn_kv_conv.weight = rows of attn_k_conv, attn_v_conv
# the bytes of every row are copied unchanged; the separate tensors are dropped
import sys, numpy as np
sys.path.insert(0, "/home/hxcliff/src/llama.cpp-r9700/gguf-py")
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
arch = None
def field_value(f):
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        sub = f.types[-1]
        if sub == GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode("utf-8") for i in f.data], sub
        return [f.parts[i].tolist()[0] for i in f.data], sub
    if t == GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode("utf-8"), t
    return f.parts[f.data[0]].tolist()[0], t
kvs = {f.name: field_value(f) for f in r.fields.values() if not f.name.startswith("GGUF.")}
arch = kvs["general.architecture"][0]
w = GGUFWriter(dst, arch=arch, endianess=r.endianess)
for k, (v, t) in kvs.items():
    if k in ("general.architecture", "general.alignment"):
        continue
    if isinstance(v, list):
        w.add_key_value(k, v, GGUFValueType.ARRAY, sub_type=t)
    else:
        w.add_key_value(k, v, t)

tens = {t.name: t for t in r.tensors}
QGRKV = ["attn_q", "attn_gate", "attn_rel", "attn_k", "attn_v"]
KVC   = ["attn_k_conv", "attn_v_conv"]
merged_from = {}
out = []
for t in r.tensors:
    p = t.name.split(".")
    if p[0] == "blk" and p[2] in QGRKV + KVC and p[3] == "weight":
        group = QGRKV if p[2] in QGRKV else KVC
        if p[2] != group[0]:
            continue
        parts = [tens[f"blk.{p[1]}.{n}.weight"] for n in group]
        ty = parts[0].tensor_type
        assert all(x.tensor_type == ty for x in parts), [x.tensor_type for x in parts]
        ne0 = int(parts[0].shape[0])
        assert all(int(x.shape[0]) == ne0 for x in parts)
        rows = sum(int(x.shape[1]) for x in parts)
        data = np.concatenate([np.asarray(x.data).reshape(int(x.shape[1]), -1) for x in parts], axis=0)
        name = f"blk.{p[1]}.{'attn_qgrkv' if group is QGRKV else 'attn_kv_conv'}.weight"
        out.append((name, data, ty, (rows, ne0)))
        merged_from[name] = [x.name for x in parts]
        continue
    out.append((t.name, t.data, t.tensor_type, None))

for name, data, ty, shape in out:
    # float data: its own shape; quantized data: the byte shape (rows, bytes per row), as the reader returns it
    w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ty)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for name, data, ty, shape in out:
    w.write_tensor_data(np.ascontiguousarray(data))
w.close()
print(f"wrote {len(out)} tensors ({len(merged_from)} merged) to {dst}")
