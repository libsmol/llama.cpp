# r9700 private tree: build, model files, quality gate

Build (static, Vulkan, clang, native): `build-tree.sh <tree> <build-dir> [targets]`.

Inkling 2 Mini UD-Q4_K_XL (Unsloth file, arch inkling2 + num_loops); both steps copy the tensor bytes unchanged:

    python3 convert_unsloth_inkling2.py Inkling-2-Mini-UD-Q4_K_XL.gguf Inkling-2-Mini-UD-Q4_K_XL-loop.gguf   # upstream inkling2_loop layout
    python3 merge_inkling2_attn.py Inkling-2-Mini-UD-Q4_K_XL-loop.gguf Inkling-2-Mini-UD-Q4_K_XL-r9700.gguf     # attn_qgrkv / attn_kv_conv

Quality gate (every change): `test-backend-ops -b Vulkan0`, then
`BASE=up-cpu64-tok2.bin kldfull.sh <bin> <name> -ngl 99 -b 1024 [-ub 512 | -ub 4 | -ub 1]` for the old and the new build
and `kldpair.py new:old`: the paired KLD difference must not be significantly positive (t < 2) on the batch,
verification and decode paths. `-b 1024` keeps one sequence per batch, as the server (with two the fused attention is
not used). The model is sensitive to the rounding order, so the whole-model KLD level is not a quality signal.
