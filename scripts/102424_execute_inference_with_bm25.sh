#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

inference_model=meta-llama/Llama-3.1-70B
rag_db=statpearls
inference_max_length=2048
quantize_type=4bit
task=mmlu
rag_max_length=512
top_k=1

time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type
