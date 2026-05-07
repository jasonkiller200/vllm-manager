#!/bin/bash
# vLLM Engine Auto-Start Script
# Query vLLM Manager API for enabled model, then start it
# This should run AFTER vllm-manager.service is up

set -e

MANAGER_URL="http://127.0.0.1:5555"
MAX_RETRIES=30
RETRY_DELAY=2

echo "[$(date)] Waiting for vLLM Manager to be ready..."

# Wait for Manager to be available
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s "$MANAGER_URL/api/status" >/dev/null 2>&1; then
        echo "[$(date)] vLLM Manager is ready."
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "[$(date)] ERROR: vLLM Manager not available after $MAX_RETRIES retries."
        exit 1
    fi
    sleep $RETRY_DELAY
done

# Find enabled model
echo "[$(date)] Querying enabled models..."
ENABLED_MODEL=$(curl -s "$MANAGER_URL/api/models" | python3 -c "
import sys, json
models = json.load(sys.stdin)
enabled = [m for m in models if m.get('enabled', False)]
if enabled:
    # If multiple enabled, pick the first one
    print(enabled[0]['path'])
else:
    print('')
    sys.exit(1)
")

if [ -z "$ENABLED_MODEL" ]; then
    echo "[$(date)] WARNING: No enabled model found. Starting without model."
    exit 0
fi

echo "[$(date)] Found enabled model: $ENABLED_MODEL"

# Start the model via Manager API
echo "[$(date)] Starting model via Manager API..."
RESPONSE=$(curl -s -X POST "$MANAGER_URL/api/start" \
    -H "Content-Type: application/json" \
    -d "{\"model_path\": \"$ENABLED_MODEL\"}")

echo "[$(date)] Response: $RESPONSE"

# Check if start was successful
if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('success') else 1)" 2>/dev/null; then
    echo "[$(date)] Model started successfully."
    exit 0
else
    echo "[$(date)] WARNING: Start API returned unexpected response. Checking if vLLM is running..."
    # Fallback: check if vLLM is already running
    if curl -s http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
        echo "[$(date)] vLLM Engine appears to be running on port 8001."
        exit 0
    else
        echo "[$(date)] ERROR: Failed to start vLLM Engine."
        exit 1
    fi
fi
