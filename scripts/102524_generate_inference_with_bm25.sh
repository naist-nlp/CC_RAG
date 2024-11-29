#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:3090:1
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
COMMENTOUT

time python $project/src/102524_generate_inference_with_bm25.py \
    --inference_model microsoft/Phi-3.5-mini-instruct \
    --top_k 3 \
    --rag_db statpearls \
    --task medmcqa \
    --rag_max_length 512 \
    --inference_max_length 2048 \
