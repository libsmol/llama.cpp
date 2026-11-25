#!/bin/bash
# kldfull.sh <bin-dir> <name> [llama-perplexity args]: KLD of CHUNKS (16) x 1024 tokens of EVAL against the reference logits
# BASE (written once by a reference build with --kl-divergence-base); the log is WORK/logs/kld_<name>.txt
BIN=$(realpath "$1"); NAME=$2; shift 2
WORK=${WORK:-$HOME/src/r9700-work}
mkdir -p "$WORK/logs" && cd "$WORK" || exit 1
"$BIN/llama-perplexity" -m "${MODEL:-$HOME/r9700/Qwen3.8-27B-UD-Q4_K_XL.gguf}" -fa on -ctk q8_0 -ctv q8_0 -f "${EVAL:-eval.txt}" -c 1024 \
  --chunks "${CHUNKS:-16}" --kl-divergence-base "${BASE:-base-logits.bin}" --kl-divergence "$@" > "logs/kld_$NAME.txt" 2>&1
grep -E 'Mean    KLD|Same top p' "logs/kld_$NAME.txt" | tr '\n' ' '; echo " [$NAME]"
