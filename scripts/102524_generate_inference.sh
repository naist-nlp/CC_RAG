#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=is-nlp
#SBATCH --job-name=generate-a6000
#SBATCH -o logs/slurm-%x-%j.log
#SBATCH --nodelist=elm66

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b
TASKS=("medqa" "medmcqa" "mmlu" "pubmedqa")
COMMENTOUT

inference_model=epfl-llm/meditron-70b
task=mmlu
inference_max_length=2048
quantize_type=4bit

echo "MODEL: $inference_model, TASK: $task, INFERENCE_MAX_LENGTH: $inference_max_length, QUANTIZE_TYPE: $quantize_type"

time python $project/src/102524_generate_inference.py \
    --inference_model $inference_model \
    --task $task \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

echo "Done"
