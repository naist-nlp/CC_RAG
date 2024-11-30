#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

base_model=meta-llama/Llama-3.1-70B
rag_db=textbooks
top_k=1
max_length=512
task=medmcqa

time python $project/src/102624_evaluate_results_bm25.py \
    --base_model_name $base_model \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task
