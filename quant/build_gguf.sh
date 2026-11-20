#!/usr/bin/env bash
# Track C: HF checkpoint -> GGUF at several precisions with llama.cpp.
#   LLAMA_CPP=~/llama.cpp bash quant/build_gguf.sh WeiboAI/VibeThinker-1.5B models/gguf/vibe
# Needs a built llama.cpp (cmake -B build && cmake --build build -j) and its python requirements.
set -euo pipefail
MODEL="$1"; OUT="$2"; LLAMA_CPP="${LLAMA_CPP:?set LLAMA_CPP to your llama.cpp checkout}"
mkdir -p "$(dirname "$OUT")"
if [ ! -d "$MODEL" ]; then
  MODEL_DIR="$(huggingface-cli download "$MODEL" --quiet)"   # prints the local snapshot path
else
  MODEL_DIR="$MODEL"
fi
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MODEL_DIR" --outtype f16 --outfile "$OUT-f16.gguf"
for Q in Q8_0 Q4_K_M Q3_K_M; do
  "$LLAMA_CPP/build/bin/llama-quantize" "$OUT-f16.gguf" "$OUT-${Q,,}.gguf" "$Q"
done
ls -lh "$OUT"-*.gguf
