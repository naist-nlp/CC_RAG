#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

i=1
base_model=meta-llama/Llama-3.1-8B

python $project/src/110824_evaluate_results_medmcqa.py \
    --base_model_name $base_model \
    --prompt_pattern $i
