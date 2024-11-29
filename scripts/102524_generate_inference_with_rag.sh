#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=is-nlp
#SBATCH --job-name=generate-with-rag
#SBATCH -o logs/slurm-%x-%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=slack:U06SLUE1DS8

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate


<< COMMENTOUT
MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "microsoft/Phi-3.5-mini-instruct" "google/gemma-2-2b-it" "axiong/PMC_LLaMA_13B")
RAG_MODELS=("ncbi/MedCPT-Query-Encoder" "intfloat/e5-base-v2" "facebook/contriever" "bm25")
TASKS=("medqa" "medmcqa" "mmlu" "pubmedqa")
RAGDB=("statpearls" "textbooks" "pubmed" "wikipedia")

epfl-llm/meditron-70b
COMMENTOUT

inference_model=microsoft/Phi-3.5-mini-instruct
rag_db=statpearls
task=medmcqa
inference_max_length=2048
quantize_type=none
top_k=3

rag_max_length=512
rag_model=allenai/specter
time python $project/src/102524_generate_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

rag_max_length=256
time python $project/src/102524_generate_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type
