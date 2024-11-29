#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:6000:1
#SBATCH --account=is-nlp
#SBATCH --job-name=inference-with-bm25
#SBATCH -o logs/slurm-%x-%j.log

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b
RAG_MODELS=("ncbi/MedCPT-Query-Encoder" "intfloat/e5-base-v2" "facebook/contriever" "bm25")
TASKS=("medqa" "medmcqa" "mmlu" "pubmedqa")
RAGDB=("statpearls" "texatbooks" "pubmed" "wikipedia")
COMMENTOUT

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
