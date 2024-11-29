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
# models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b

for base_model in microsoft/Phi-3.5-mini-instruct axiong/PMC_LLaMA_13B meta-llama/Llama-3.1-70B epfl-llm/meditron-70b meta-llama/Llama-2-70b-chat-hf
do
    base_model_suffix=$(echo $base_model | sed -e 's/.*\///')
    task=mmlu

    echo $base_model, $task
    python $project/src/102924_generate_table.py \
        --base_model $base_model \
        --task $task > $project/table-md/table-$base_model_suffix-$task.md

    echo table-$base_model_suffix-$task.md

    task=medqa
    python $project/src/102924_generate_table.py \
        --base_model $base_model \
        --task $task > $project/table-md/table-$base_model_suffix-$task.md

    echo table-$base_model_suffix-$task.md

    task=pubmedqa
    python $project/src/102924_generate_table.py \
        --base_model $base_model \
        --task $task > $project/table-md/table-$base_model_suffix-$task.md

    echo table-$base_model_suffix-$task.md

    task=medmcqa
    python $project/src/102924_generate_table.py \
        --base_model $base_model \
        --task $task > $project/table-md/table-$base_model_suffix-$task.md
    echo table-$base_model_suffix-$task.md
done

echo "Job finished"
