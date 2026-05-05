#!/bin/bash
set -euo pipefail

N_GPUS=${1:-1}
mkdir -p logs

# vLLM requires libcudart.so.13; expose Ollama's bundled CUDA 13 runtime.
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:${LD_LIBRARY_PATH:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

declare -A SLOT_PID

i=0
for vlm in qwen gemma deepseek kimi intern; do
    for mode in multi image text; do
        slot=$((i % N_GPUS))
        [[ -n "${SLOT_PID[$slot]+_}" ]] && wait "${SLOT_PID[$slot]}"

        echo "$(date '+%H:%M:%S')  $((i+1))/12  vlm=$vlm  mode=$mode  gpu=$slot"
        CUDA_VISIBLE_DEVICES="$slot" python methodology/run.py --vlm "$vlm" --mode "$mode" \
            > "logs/${vlm}_${mode}.log" 2>&1 &
        SLOT_PID[$slot]=$!
        i=$((i+1))
    done
done

for slot in "${!SLOT_PID[@]}"; do wait "${SLOT_PID[$slot]}"; done
echo "Done."
