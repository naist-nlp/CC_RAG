#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:3090:1
#SBATCH --account=is-nlp
#SBATCH --job-name=manipulation-3090
#SBATCH -o logs/slurm-%x-%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=slack:U06SLUE1DS8

project=$(pwd)
source $project/.venv/bin/activate

# models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b meta-llama/Llama-2-70b-chat-hf
#prompt_patternは1,2,3

for i in 1 2 3
do
for model in meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-70B epfl-llm/meditron-70b axiong/PMC_LLaMA_13B microsoft/Phi-3.5-mini-instruct meta-llama/Llama-2-70b-chat-hf
do
    python $project/src/110824_evaluate_results_medmcqa.py \
    --base_model_name $model \
    --prompt_pattern $i
done
done

echo "Done"
