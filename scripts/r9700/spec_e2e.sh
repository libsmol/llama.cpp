#!/bin/bash
# spec_e2e.sh <bin-dir> <tag> [server args]: start llama-server on :18080, run spec_client.py, stop it
BIN=$1; TAG=$2; shift; shift
W=$HOME/src/inkling-work
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 "$BIN/llama-server" -m ${SPEC_MODEL:-$HOME/Inkling-2-Mini-UD-Q4_K_XL-r9700.gguf} -c 32768 -np 1 -ngl all -fa on -ctk q8_0 -ctv q8_0 \
  --cache-ram 0 -ub 512 --host 127.0.0.1 --port 18080 "$@" < /dev/null > $W/logs/server-$TAG.log 2>&1 &
PID=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:18080/health > /dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
if curl -sf http://127.0.0.1:18080/health > /dev/null 2>&1; then
  python3 $W/spec_client.py $TAG
else
  echo "server $TAG failed to start"; tail -15 $W/logs/server-$TAG.log
fi
kill $PID 2>/dev/null; wait $PID 2>/dev/null
