#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

base_model=microsoft/Phi-3.5-mini-instruct
task=mmlu

python $project/src/110624_plot_two_variable.py \
    --base_model $base_model \
    --task $task
