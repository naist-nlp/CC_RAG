#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

inference_model=meta-llama/Llama-3.1-70B-Instruct
quantize_type=none
prompt_pattern=1

time python $project/src/110824_execute_inference_medmcqa_rag_other3.py \
    --inference_model $inference_model \
    --quantize_type $quantize_type \
    --prompt_pattern $prompt_pattern
