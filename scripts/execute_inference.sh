#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

inference_model=meta-llama/Llama-3.1-70B-Instruct
quantize_type=none
inference_max_length=2048
task=medmcqa

time python $project/src/102424_execute_inference.py \
    --inference_model $inference_model \
    --task $task \
    --quantize_type $quantize_type \
    --inference_max_length $inference_max_length
