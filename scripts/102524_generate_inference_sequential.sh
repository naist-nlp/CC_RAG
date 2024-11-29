#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=is-nlp
#SBATCH --job-name=generate-v100
#SBATCH -o logs/slurm-%x-%j.log
#SBATCH --nodelist=elm41

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
MODELS=("microsoft/Phi-3.5-mini-instruct" "axiong/PMC_LLaMA_13B")
TASKS=("medqa" "medmcqa" "mmlu" "pubmedqa")
COMMENTOUT

inference_model=axiong/PMC_LLaMA_13B

inference_max_length=2048
quantize_type=8bit

task=pubmedqa
echo "MODEL: $inference_model, TASK: $task, INFERENCE_MAX_LENGTH: $inference_max_length, QUANTIZE_TYPE: $quantize_type"
time python $project/src/102524_generate_inference.py \
    --inference_model $inference_model \
    --task $task \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

echo "Done"
