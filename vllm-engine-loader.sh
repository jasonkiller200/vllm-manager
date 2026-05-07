#!/bin/bash
# vLLM Engine Auto-Load Script
# Queries vLLM Manager for enabled models and starts them via API

set -euo pipefail

MANAGER_URL="http://127.0.0.1:5555"
MAX_WAIT=60
ELAPSED=0

echo "[vllm-loader] Waiting for vLLM Manager at $MANAGER_URL ..."

while ! curl -sf "$MANAGER_URL/api/status" >/dev/null 2>&1; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "[vllm-loader] ERROR: Manager not reachable after ${MAX_WAIT}s"
        exit 1
    fi
done

echo "[vllm-loader] Manager ready (${ELAPSED}s)"

# Find enabled models
ENABLED=$(curl -sf "$MANAGER_URL/api/models" | python3 -c "
import sys, json
models = json.load(sys.stdin)
for m in models:
    if m.get('enabled'):
        print(m['path'])
")

if [ -z "$ENABLED" ]; then
    echo "[vllm-loader] No enabled models found, nothing to start."
    exit 0
fi

echo "[vllm-loader] Starting enabled model(s):"
echo "$ENABLED" | while IFS= read -r model_path; do
    echo "[vllm-loader]   → $model_path"
    RESPONSE=$(curl -sf -X POST "$MANAGER_URL/api/start" \
        -H "Content-Type: application/json" \
        -d "{\"model_path\": \"$model_path\"}" 2>&1) || true
    echo "[vllm-loader]   API response: $RESPONSE"
done

echo "[vllm-loader] Done."
