#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

base_model=microsoft/Phi-3.5-mini-instruct
task=medmcqa

python $project/src/110524_plot_calibration_curve.py \
    --model $base_model \
    --task $task

