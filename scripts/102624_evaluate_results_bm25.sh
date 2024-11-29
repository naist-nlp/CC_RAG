#!/bin/bash
#SBATCH -p lang_long
#SBATCH -c 8
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=evaluate-bm25
#SBATCH -o logs/slurm-%x-%j.log
set -eu

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate


<< COMMENTOUT
db_path: pubmed, statperarls, textbooks, wikipedia
models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b  meta-llama/Llama-2-70b-chat-hf
COMMENTOUT


basemodels=(
    microsoft/Phi-3.5-mini-instruct
    axiong/PMC_LLaMA_13B
    meta-llama/Llama-3.1-70B
    epfl-llm/meditron-70b
    meta-llama/Llama-2-70b-chat-hf
)
rag_dbs=(
    statpearls
    textbooks
)
top_ks=(
    1
    3
)
max_lengths=(
    256
    512
)

for base_model_name in ${basemodels[@]}; do
    for rag_db in ${rag_dbs[@]}; do
        for top_k in ${top_ks[@]}; do
            for max_length in ${max_lengths[@]}; do
                echo "base_model_name: $base_model_name, rag_db: $rag_db, top_k: $top_k, max_length: $max_length"
                task=medmcqa
                time python $project/src/102624_evaluate_results_bm25.py \
                    --base_model_name $base_model_name \
                    --rag_db $rag_db \
                    --top_k $top_k \
                    --max_length $max_length \
                    --task $task

                task=medqa
                time python $project/src/102624_evaluate_results_bm25.py \
                    --base_model_name $base_model_name \
                    --rag_db $rag_db \
                    --top_k $top_k \
                    --max_length $max_length \
                    --task $task

                task=pubmedqa
                time python $project/src/102624_evaluate_results_bm25.py \
                    --base_model_name $base_model_name \
                    --rag_db $rag_db \
                    --top_k $top_k \
                    --max_length $max_length \
                    --task $task

                task=mmlu
                time python $project/src/102624_evaluate_results_bm25.py \
                    --base_model_name $base_model_name \
                    --rag_db $rag_db \
                    --top_k $top_k \
                    --max_length $max_length \
                    --task $task
            done
        done
    done
done

