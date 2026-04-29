#!/bin/bash
set -euo pipefail

N_GPUS=${1:-1}
mkdir -p logs

declare -A SLOT_PID

i=0
for vlm in qwen gemma hunyuan kimi; do
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
