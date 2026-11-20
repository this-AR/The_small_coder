#!/usr/bin/env bash
# Track C: serve one GGUF with llama-server and run the harness against it, then stop the server.
#   LLAMA_CPP=~/llama.cpp bash quant/serve_and_eval.sh models/gguf/vibe-q4_k_m.gguf configs/quant_vibe.yaml q4_k_m
set -euo pipefail
GGUF="$1"; CFG="$2"; TAG="$3"; PORT="${PORT:-8080}"; NP="${NP:-8}"
"$LLAMA_CPP/build/bin/llama-server" -m "$GGUF" -c $((12288 * NP)) -np "$NP" --port "$PORT" -ngl 99 &
PID=$!
trap 'kill $PID' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 2; done
# fill the precision tag into name/out so every precision lands in its own rows
sed "s/{TAG}/$TAG/g" "$CFG" > "/tmp/quant_$TAG.yaml"
python scripts/run_eval.py "/tmp/quant_$TAG.yaml"
