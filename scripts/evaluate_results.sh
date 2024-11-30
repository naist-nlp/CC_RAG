#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

base_model=microsoft/Phi-3.5-mini-instruct
task=medqa

time python $project/src/102624_evaluate_results.py \
    --base_model_name $base_model_name \
    --task $task
