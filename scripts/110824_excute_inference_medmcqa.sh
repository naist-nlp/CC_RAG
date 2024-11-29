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

time python $project/src/110824_execute_inference_medmcqa.py \
    --inference_model microsoft/Phi-3.5-mini-instruct \
    --quantize_type none


echo "Done"
