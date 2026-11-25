# r9700: Qwen on the AMD Radeon AI PRO R9700 (RDNA4, Vulkan)

Build (static, Vulkan, clang, native): `build-tree.sh <tree> <build-dir> [targets]`.

Serving (one slot, Qwen3.8-27B with a DSpine draft, q8_0 KV cache, compute on the graphics queue):

    GGML_VK_ALLOW_GRAPHICS_QUEUE=1 llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf -md Qwen3.8-27B-DSpine-b16-Q8_0.gguf \
        -np 1 -ngl all -ngld all -fa on -ctk q8_0 -ctv q8_0 -ub 1024 \
        --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-sampling probabilistic --backend-sampling

Quality gate (every change): `test-backend-ops -b Vulkan0`, then for the old and the new build
`kldfull.sh <bin-dir> <name> -ngl 99 -b 1024 [-ub 1024 | -ub 8 | -ub 1]` and `kldpair.py new:old`: the paired KLD difference
must not be significantly positive (t < 2) on the prefill, verification and decode paths. The reference logits
(`BASE`, default `base-logits.bin` in `WORK`) are written once by a reference build with `--kl-divergence-base`;
`MODEL`, `EVAL` and `CHUNKS` select the model, the text and its length.
