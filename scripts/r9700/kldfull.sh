#!/bin/bash
# usage: kldfull.sh <bin-dir> <name> [extra args] -- CHUNKS (default 16) x 1024 vs the CPU reference BASE, full log in logs/kld_<name>.txt
BIN=$1; NAME=$2; shift; shift
cd ~/src/inkling-work
$BIN/llama-perplexity -m ${MODEL:-$HOME/Inkling-2-Mini-UD-Q4_K_XL-loop.gguf} -fa on -ctk q8_0 -ctv q8_0 -f eval.txt -c 1024 --chunks ${CHUNKS:-16} \
  --kl-divergence-base ${BASE:-up-cpu64-tok2.bin} --kl-divergence "$@" > logs/kld_$NAME.txt 2>&1
grep -E 'Mean    KLD|Same top p' logs/kld_$NAME.txt | tr '\n' ' '; echo " [$NAME]"
