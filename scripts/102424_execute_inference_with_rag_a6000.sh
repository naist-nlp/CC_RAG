#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:6000:1
#SBATCH --account=is-nlp
#SBATCH --job-name=inference-with-rag-6000ada
#SBATCH -o logs/slurm-%x-%j.log

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate


<< COMMENTOUT
db_path: pubmed, statperarls, textbooks, wikipedia
rag_model: ncbi/MedCPT-Query-Encoder, allenai/specter, facebook/contriever
models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b
RAGDB=("statpearls" "textbooks" "pubmed" "wikipedia")
task=("medqa" "medmcqa" "mmlu" "pubmedqa")
COMMENTOUT

inference_model=epfl-llm/meditron-70b
quantize_type=4bit
rag_db=statpearls
task=medqa
inference_max_length=2048

rag_model=allenai/specter
rag_max_length=512
top_k=3
time python $project/src/102424_execute_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --top_k $top_k \
    --rag_db $rag_db \
    --task $task \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

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

rag_max_length=256
top_k=3
time python $project/src/102424_execute_inference_with_rag.py \
    --inference_model $inference_model \
    --rag_model $rag_model \
    --top_k $top_k \
    --rag_db $rag_db \
    --task $task \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

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

#####################################################
