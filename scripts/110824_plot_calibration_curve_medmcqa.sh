#!/bin/bash
#SBATCH -p lang_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=plot-calibration-curve
#SBATCH -o logs/slurm-%x-%j.log
set -eu

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

# task: pubmedqa, medmcqa, medqa, mmlu
# models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b meta-llama/Llama-2-70b-chat-hf

for base_model in microsoft/Phi-3.5-mini-instruct axiong/PMC_LLaMA_13B meta-llama/Llama-3.1-70B epfl-llm/meditron-70b meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.1-8B
do
    base_model_suffix=$(echo $base_model | sed -e 's/.*\///')
    python $project/src/110824_plot_calibration_curve_medmcqa.py \
        --model $base_model
    echo Done $base_model
done

echo "Job finished"
