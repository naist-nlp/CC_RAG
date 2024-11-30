#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

inference_model=meta-llama/Llama-3.1-8B
quantize_type=half
prompt_pattern=1

time python $project/src/110824_execute_inference_medmcqa_rag_ans1_other2.py \
    --inference_model $inference_model \
    --quantize_type $quantize_type \
    --prompt_pattern $prompt_pattern
