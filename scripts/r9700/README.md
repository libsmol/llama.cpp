# r9700 private tree: model files and quality gates

Inkling 2 Mini UD-Q4_K_XL (Unsloth file, arch inkling2 + num_loops):

    python3 convert_unsloth_inkling2.py Inkling-2-Mini-UD-Q4_K_XL.gguf Inkling-2-Mini-UD-Q4_K_XL-loop.gguf   # upstream inkling2_loop layout
    python3 verify_convert.py Inkling-2-Mini-UD-Q4_K_XL.gguf Inkling-2-Mini-UD-Q4_K_XL-loop.gguf official_names.json
    python3 merge_inkling2_attn.py Inkling-2-Mini-UD-Q4_K_XL-loop.gguf Inkling-2-Mini-UD-Q4_K_XL-r9700.gguf     # attn_qgrkv / attn_kv_conv

All three steps copy the tensor bytes unchanged.

Quality gate (every change): test-backend-ops -b Vulkan0, test-inkling-fused, then
`BASE=up-cpu64-tok2.bin kldfull.sh <bin> <name> -ngl 99 -b 1024 [-ub 512 | -ub 4 | -ub 1]` (one sequence per batch, as the server: with two the fused attention is not used; the KLD mode takes the chunk count of the base file) and `kldpair.py new:old`:
the paired KLD difference must not be significantly positive (t < 2) on the batch, verification and decode paths.
The model is sensitive to the rounding order (f32 FA on/off alone gives KLD ~0.002), so the whole-model KLD level is not a quality signal.
