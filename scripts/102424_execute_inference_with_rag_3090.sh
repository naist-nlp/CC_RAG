#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:3090:1
#SBATCH --account=is-nlp
#SBATCH --job-name=inference-with-rag
#SBATCH -o logs/slurm-%x-%j.log

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
MODELS=("microsoft/Phi-3.5-mini-instruct" "axiong/PMC_LLaMA_13B")
RAG_MODELS=("ncbi/MedCPT-Query-Encoder" "facebook/contriever" "allenai/specter")
TASKS=("medqa" "medmcqa" "mmlu" "pubmedqa")
RAGDB=("statpearls" "textbooks" "pubmed" "wikipedia")
COMMENTOUT

inference_model=axiong/PMC_LLaMA_13B
rag_model=facebook/contriever
rag_db=textbooks
task=medqa
quantize_type=none


time python $project/src/102424_execute_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --top_k 3 \
    --rag_db $rag_db \
    --task $task \
    --quantize_type $quantize_type \
    --rag_max_length 512 \
    --inference_max_length 2048 \


time python $project/src/102424_execute_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --top_k 3 \
    --rag_db $rag_db \
    --task $task \
    --quantize_type $quantize_type \
    --rag_max_length 256 \
    --inference_max_length 2048 \

time python $project/src/102424_execute_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --top_k 1 \
    --rag_db $rag_db \
    --task $task \
    --quantize_type $quantize_type \
    --rag_max_length 512 \
    --inference_max_length 2048 \

time python $project/src/102424_execute_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --top_k 1 \
    --rag_db $rag_db \
    --quantize_type $quantize_type \
    --task $task \
    --rag_max_length 256 \
    --inference_max_length 2048 \
