#!/bin/bash
#
# BudgetVID on Qwen2.5-VL. Mirrors scripts/qwen2_5_vl.sh (FlashVID) parameter
# for parameter, so the two logs are directly comparable.
#
# Parity check before trusting any new policy: this script with
# ALLOCATION=uniform must reproduce scripts/qwen2_5_vl.sh to the decimal.
# Narrow it to one setting to keep that cheap, e.g.
#
#   ALLOCATION=uniform TASKS=videomme RETENTION_RATIOS=0.25 bash scripts/budgetvid/qwen2_5_vl.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Evaluation benchmarks.
# Note: MVBench's action_antonym labels are ~50% wrong against the videos; do
# not put that subtask in a results table.
read -r -a TASKS <<< "${TASKS:-videomme egoschema mvbench longvideobench_val_v mlvu_test}"

# Pretrained model path.
PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"

# ! BudgetVID arguments.
ALLOCATION=${ALLOCATION:-uniform}   # see budgetvid/allocation.py
ENFORCE_BUDGET=${ENFORCE_BUDGET:-True}
read -r -a RETENTION_RATIOS <<< "${RETENTION_RATIOS:-0.10 0.15 0.20 0.25}"

# ! Inherited FlashVid arguments. Keep these identical to scripts/qwen2_5_vl.sh.
## Dyseg (fixed)
DO_SEGMENT=True
MIN_SEGMENT_NUM=4
COMPLEMENTARY_SEGMENT=True
## ADTS and TSTM (fixed)
TOKEN_SELECTION_METHOD=attn_div # * Use ADTSv1 for Qwen2.5-VL
ALPHA=0.70
TEMPORAL_THRESHOLD=0.8
## Inner-LLM Pruning (fixed)
EXPANSION=1.25
PRUNING_LAYER=20
LLM_RETENTION_RATIO=0.3

BASE_BUDGETVID_ARGS="enable_budgetvid=True,allocation=$ALLOCATION,enforce_budget=$ENFORCE_BUDGET,expansion=$EXPANSION,do_segment=$DO_SEGMENT,min_segment_num=$MIN_SEGMENT_NUM,complementary_segment=$COMPLEMENTARY_SEGMENT,token_selection_method=$TOKEN_SELECTION_METHOD,alpha=$ALPHA,temporal_threshold=$TEMPORAL_THRESHOLD,pruning_layer=$PRUNING_LAYER,llm_retention_ratio=$LLM_RETENTION_RATIO"

# Model arguments.
MAX_NUM_FRAMES=32
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running allocation=${ALLOCATION} with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,$BASE_BUDGETVID_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        accelerate launch \
        --main_process_port 18888 \
        --num_processes 8 \
        -m lmms_eval \
        --model qwen2_5_vl \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "qwen2_5_vl" \
        --output_path ./logs/budgetvid/${ALLOCATION}
    done
    echo "Finished allocation=${ALLOCATION} with retention_ratio=${retention_ratio}"
done
