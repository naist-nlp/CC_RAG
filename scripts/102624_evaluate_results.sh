#!/bin/bash
#SBATCH -p lang_long
#SBATCH -c 8
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=evaluate
#SBATCH -o logs/slurm-%x-%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=slack:U06SLUE1DS8
set -eu

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b  meta-llama/Llama-2-70b-chat-hf, meta-llama/Llama-3.1-8B
COMMENTOUT


for base_model_name in microsoft/Phi-3.5-mini-instruct axiong/PMC_LLaMA_13B meta-llama/Llama-3.1-70B epfl-llm/meditron-70b meta-llama/Llama-2-70b-chat-hf
do
    task=medqa
    time python $project/src/102624_evaluate_results.py \
        --base_model_name $base_model_name \
        --task $task

    task=medmcqa
    time python $project/src/102624_evaluate_results.py \
        --base_model_name $base_model_name \
        --task $task

    task=mmlu
    time python $project/src/102624_evaluate_results.py \
        --base_model_name $base_model_name \
        --task $task

    task=pubmedqa
    time python $project/src/102624_evaluate_results.py \
        --base_model_name $base_model_name \
        --task $task
done

echo "Job finished"
