#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:6000:1
#SBATCH --account=is-nlp
#SBATCH --job-name=manipulation-6000
#SBATCH -o logs/slurm-%x-%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=slack:U06SLUE1DS8

project=$(pwd)
source $project/.venv/bin/activate

# models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b meta-llama/Llama-2-70b
# meta-llama/Llama-3.1-8B

time python $project/src/110824_execute_inference_medmcqa_rag_ans1.py \
    --inference_model meta-llama/Llama-3.1-8B \
    --quantize_type half \
    --prompt_pattern 1



echo "Done"
