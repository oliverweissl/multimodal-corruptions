#!/bin/bash
# Worker-pool pattern: N GPU workers each pull VLMs from a shared queue.
# As soon as a slot finishes, the next VLM starts immediately.
#
# Usage: bash 03_run_all.sh [N_GPUS]   (default: 1)
set -euo pipefail

N_GPUS=${1:-1}
mkdir -p logs

export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:${LD_LIBRARY_PATH:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

declare -A VLM_MODEL_IDS=(
    ["qwen"]="Qwen/Qwen3-VL-4B-Instruct"
    ["gemma"]="google/gemma-3-4b-it"
    ["kimi"]="moonshotai/Kimi-VL-A3B-Instruct"
    ["deepseek"]="deepseek-ai/deepseek-vl2-tiny"
    ["intern"]="OpenGVLab/InternVL3_5-8B"
)

VLMS=(qwen gemma deepseek kimi intern)

# Shared queue state
QUEUE_IDX_FILE="/tmp/vlm_run_queue_idx_$$"
QUEUE_LOCK_FILE="/tmp/vlm_run_queue_lock_$$"
echo 0 > "$QUEUE_IDX_FILE"
cleanup() { rm -f "$QUEUE_IDX_FILE" "$QUEUE_LOCK_FILE"; kill "$EMB_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Pop the next VLM name from the queue (thread-safe via flock)
pop_vlm() {
    local vlm=""
    (
        flock -x 9
        local idx
        idx=$(cat "$QUEUE_IDX_FILE")
        if [ "$idx" -lt "${#VLMS[@]}" ]; then
            vlm="${VLMS[$idx]}"
            echo $((idx + 1)) > "$QUEUE_IDX_FILE"
        fi
        echo "$vlm"
    ) 9>"$QUEUE_LOCK_FILE"
}

# Worker: one per GPU slot, keeps serving VLMs until the queue is empty
worker() {
    local gpu=$1
    local port=$2

    while true; do
        local vlm
        vlm=$(pop_vlm)
        [ -z "$vlm" ] && break

        local model_id="${VLM_MODEL_IDS[$vlm]}"
        echo "$(date '+%H:%M:%S')  [GPU $gpu] Serving $vlm ($model_id) on port $port"

        fuser -k "${port}/tcp" 2>/dev/null || true; sleep 1

        CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$model_id" \
            --port "$port" \
            --gpu-memory-utilization 0.85 \
            --enforce-eager --trust-remote-code --max-model-len 4096 \
            > "logs/${vlm}_server.log" 2>&1 &
        local server_pid=$!

        until curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
            sleep 3
            kill -0 "$server_pid" 2>/dev/null || { echo "ERROR: $vlm server died"; break 2; }
        done
        echo "$(date '+%H:%M:%S')  [GPU $gpu] $vlm ready — running modes sequentially"

        for mode in multi image text; do
            echo "$(date '+%H:%M:%S')  [GPU $gpu] $vlm/$mode"
            CUDA_VISIBLE_DEVICES="$gpu" python methodology/run.py \
                --vlm "$vlm" --mode "$mode" \
                > "logs/${vlm}_${mode}.log" 2>&1
        done

        echo "$(date '+%H:%M:%S')  [GPU $gpu] $vlm done — stopping server"
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    done
    echo "$(date '+%H:%M:%S')  [GPU $gpu] queue empty, worker done"
}

# --- Start shared embedding server ---
echo "$(date '+%H:%M:%S')  Starting embedding server on GPU 0 port 8699"
fuser -k 8699/tcp 2>/dev/null || true; sleep 1
CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen3-Embedding-0.6B" \
    --port 8699 --task embed \
    --gpu-memory-utilization 0.05 \
    --enforce-eager --trust-remote-code --max-model-len 4096 \
    > logs/embedding_server.log 2>&1 &
EMB_PID=$!
until curl -sf http://localhost:8699/health > /dev/null 2>&1; do
    sleep 3
    kill -0 "$EMB_PID" 2>/dev/null || { echo "ERROR: embedding server died"; exit 1; }
done
echo "$(date '+%H:%M:%S')  Embedding server ready"

# --- Launch N GPU workers in parallel ---
for (( slot=0; slot<N_GPUS; slot++ )); do
    worker "$slot" "$((8700 + slot))" &
done
wait

echo "$(date '+%H:%M:%S')  All done."
