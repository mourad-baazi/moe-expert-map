#!/usr/bin/env bash
# Starts Inkling 2 Mini with loop-aware expert placement and a chat page at http://localhost:8080
#   ./chat.sh        8K context,  8-bit context memory (practically lossless)   ~18-19 tokens/s on RTX 2060 6 GB
#   ./chat.sh long   64K context, 4-bit context memory (tiny quality cost)     ~17-19 tokens/s
# Both keep the experts of 10 looped layers (8-17) on the GPU; 11+ overflows 6 GB inside llama-server.
# Stop with Ctrl+C.
cd "$(dirname "$0")"
# Ctrl+C in Git Bash can leave llama-server running and holding the port; stop any old one first.
taskkill //IM llama-server.exe //F > /dev/null 2>&1 && echo "Stopped an old llama-server that was still running."

if [ "$1" = "long" ]; then
  CTX=65536; LAYERS='[8-9]|1[0-7]'; KV=q4_0
else
  CTX=8192;  LAYERS='[8-9]|1[0-7]'; KV=q8_0
fi
echo "Context: $CTX tokens. Loading the model... when you see 'server is listening', open http://localhost:8080"

exec ./toolchain/llama-cuda/llama-server.exe \
  -m models/Inkling-2-Mini-UD-Q2_K_XL-mainline.gguf \
  -ngl 99 \
  -ot "blk\.($LAYERS)\.ffn_.*_exps=CUDA0,exps=CPU" \
  -t 6 \
  -c $CTX \
  -np 1 \
  -ub 128 -b 128 \
  -fa on -ctk $KV -ctv $KV \
  --jinja \
  --chat-template-file templates/inkling2-chat-fixed.jinja \
  --host 127.0.0.1 --port 8080
