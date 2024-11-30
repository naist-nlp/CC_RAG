#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

base_model=microsoft/Phi-3.5-mini-instruct
python $project/src/110824_plot_two_variable_medmcqa.py \
    --base_model $base_model
