#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

base_model_name=microsoft/Phi-3.5-mini-instruct
rag_db=textbooks
task=medqa
rag_model_name=facebook/contriever
max_length=512
top_k=3

time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task
