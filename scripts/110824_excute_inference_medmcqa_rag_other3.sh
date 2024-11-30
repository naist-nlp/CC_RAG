#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:3090:1
#SBATCH --account=is-nlp
#SBATCH --job-name=manipulation-3090
#SBATCH -o logs/slurm-%x-%j.log

project=$(pwd)
source $project/.venv/bin/activate

inference_model=meta-llama/Llama-3.1-70B-Instruct
quantize_type=none
prompt_pattern=1

time python $project/src/110824_execute_inference_medmcqa_rag_other3.py \
    --inference_model $inference_model \
    --quantize_type $quantize_type \
    --prompt_pattern $prompt_pattern
