#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=is-nlp
#SBATCH --job-name=inference-llama3.1-instruct
#SBATCH -o logs/slurm-%x-%j.log
#SBATCH --nodelist=elm66

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
MODELS=("microsoft/Phi-3.5-mini-instruct" "axiong/PMC_LLaMA_13B" epfl-llm/meditron-70b meta-llama/Llama-3.1-70B meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.1-8B
task=("medqa" "medmcqa" "mmlu" "pubmed")
COMMENTOUT

inference_model=meta-llama/Llama-3.1-70B-Instruct
quantize_type=none
inference_max_length=2048

task=medmcqa
time python $project/src/102424_execute_inference.py \
    --inference_model $inference_model \
    --task $task \
    --quantize_type $quantize_type \
    --inference_max_length $inference_max_length

task=mmlu
time python $project/src/102424_execute_inference.py \
    --inference_model $inference_model \
    --task $task \
    --quantize_type $quantize_type \
    --inference_max_length $inference_max_length

task=pubmedqa
time python $project/src/102424_execute_inference.py \
    --inference_model $inference_model \
    --task $task \
    --quantize_type $quantize_type \
    --inference_max_length $inference_max_length

task=medqa
time python $project/src/102424_execute_inference.py \
    --inference_model $inference_model \
    --task $task \
    --quantize_type $quantize_type \
    --inference_max_length $inference_max_length

echo "Done"
