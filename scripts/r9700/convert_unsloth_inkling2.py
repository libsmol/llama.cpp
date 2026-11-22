#!/usr/bin/env python3
# convert_unsloth_inkling2.py <src.gguf> <dst.gguf>
# Unsloth's looped Inkling 2 GGUF (arch inkling2 + num_loops, 48 virtual blocks, global loop query tables)
# -> the upstream inkling2_loop layout (24 blocks, per-layer loop_msg_query, loop_*_ctrl_* names).
# Tensor data is copied byte for byte; the two query tables are re-laid out exactly:
#   blk.{s+i}.loop_msg_query[:, q, k, v, ffn] = res[2i], kv[2i], kv[2i+1], res[2i+1]   (i = 0 .. core_end - core_start - 1)
import sys, numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "../../gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
A0, A1 = "inkling2", "inkling2_loop"

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
assert kvs["general.architecture"][0] == A0, kvs["general.architecture"]
n_virt  = kvs[f"{A0}.block_count"][0]
n_loops = kvs[f"{A0}.num_loops"][0]
cs, ce  = kvs[f"{A0}.loop.core_start"][0], kvs[f"{A0}.loop.core_end"][0]
n_phys  = n_virt - (n_loops - 1)*(ce - cs)
print(f"virtual blocks {n_virt}, loops {n_loops}, core [{cs}, {ce}) -> {n_phys} physical blocks")

w = GGUFWriter(dst, arch=A1, endianess=r.endianess)
rename_kv = {f"{A0}.num_loops": f"{A1}.loop.count", f"{A0}.loop.core_start": f"{A1}.loop.start",
             f"{A0}.loop.core_end": f"{A1}.loop.end", f"{A0}.loop.gate_rank": f"{A1}.loop.gate_rank"}
for k, (v, t) in kvs.items():
    if k in ("general.architecture", "general.alignment"):
        continue
    nk = rename_kv.get(k, A1 + k[len(A0):] if k.startswith(A0 + ".") else k)
    if k == f"{A0}.block_count":
        v = n_phys
    if isinstance(v, list) and k.startswith(A0 + ".") and len(v) == n_virt:
        assert v[n_phys:] == [v[(i - n_phys) % (ce - cs) + cs] for i in range(n_phys, n_virt)], k
        v = v[:n_phys]
    if isinstance(v, list):
        w.add_key_value(nk, v, GGUFValueType.ARRAY, sub_type=t)
    else:
        w.add_key_value(nk, v, t)
w.add_bool(f"{A1}.loop.varying", False)

ren = {"attn": "q", "k": "k", "v": "v", "mlp": "ffn"}
tens = {t.name: t for t in r.tensors}
res, kvq = tens["loop_res_query.weight"], tens["loop_kv_res_query.weight"]
res_d = np.asarray(res.data).reshape(-1, 2304); kv_d = np.asarray(kvq.data).reshape(-1, 2304)
assert res_d.shape[0] == 2*(ce - cs) and kv_d.shape[0] == 2*(ce - cs)

out = []   # (name, data ndarray, tensor_type)
for t in r.tensors:
    name = t.name
    if name in ("loop_res_query.weight", "loop_kv_res_query.weight"):
        continue
    parts = name.split(".")
    if parts[0] == "blk" and parts[2].startswith("loop_"):
        mod, kind = parts[2][5:].rsplit("_", 1)
        name = f"blk.{parts[1]}.loop_{ren[mod]}_ctrl_{kind}.weight"
    out.append((name, t))
    if parts[0] == "blk" and parts[2] == "ffn_post_conv":
        il = int(parts[1])
        if cs <= il < ce:
            i = il - cs
            mq = np.stack([res_d[2*i], kv_d[2*i], kv_d[2*i + 1], res_d[2*i + 1]]).astype(np.float32)   # [4, 2304]
            out.append((f"blk.{il}.loop_msg_query.weight", mq))

for name, t in out:
    if isinstance(t, np.ndarray):
        w.add_tensor_info(name, t.shape, t.dtype, t.nbytes, gguf.GGMLQuantizationType.F32)
    else:
        w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for name, t in out:
    w.write_tensor_data(t if isinstance(t, np.ndarray) else t.data)
w.close()
print(f"wrote {len(out)} tensors to {dst}")
