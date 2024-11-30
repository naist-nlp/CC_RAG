#!/bin/bash
set -eu

project=$(pwd)
source $project/.venv/bin/activate

task=medmcqa
rag_db=textbooks
rag_model=ncbi/MedCPT-Query-Encoder
inference_model=microsoft/Phi-3.5-mini-instruct

quantize_type=none
inference_max_length=2048
rag_max_length=512
top_k=1

time python $project/src/102424_execute_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --top_k $top_k \
    --rag_db $rag_db \
    --task $task \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type
