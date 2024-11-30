#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

base_model=microsoft/Phi-3.5-mini-instruct

python $project/src/110824_plot_calibration_curve_medmcqa.py \
    --model $base_model
