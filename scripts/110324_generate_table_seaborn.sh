#!/bin/bash
#SBATCH -p lang_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=generate-table
#SBATCH -o logs/slurm-%x-%j.log
set -eu

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

# task: pubmedqa, medmcqa, medqa, mmlu
# models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b meta-llama/Llama-2-70b-chat-hf

base_model=meta-llama/Llama-3.1-70B

for base_model in microsoft/Phi-3.5-mini-instruct axiong/PMC_LLaMA_13B meta-llama/Llama-3.1-70B epfl-llm/meditron-70b meta-llama/Llama-2-70b-chat-hf
do
    base_model_suffix=$(echo $base_model | sed -e 's/.*\///')
    task=mmlu
    python $project/src/110324_generate_table_seaborn.py \
        --base_model $base_model \
        --task $task

    task=medqa
    python $project/src/110324_generate_table_seaborn.py \
        --base_model $base_model \
        --task $task

    task=pubmedqa
    python $project/src/110324_generate_table_seaborn.py \
        --base_model $base_model \
        --task $task

    task=medmcqa
    python $project/src/110324_generate_table_seaborn.py \
        --base_model $base_model \
        --task $task
done

echo "Job finished"
