#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

inference_model=microsoft/Phi-3.5-mini-instruct
quantize_type=none

time python $project/src/110824_execute_inference_medmcqa.py \
    --inference_model $inference_model \
    --quantize_type $quantize_type
