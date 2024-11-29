#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:3090:1
#SBATCH --account=is-nlp
#SBATCH --job-name=inference
#SBATCH -o logs/slurm-%x-%j.log

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
MODELS=("microsoft/Phi-3.5-mini-instruct" "axiong/PMC_LLaMA_13B" epfl-llm/meditron-70b meta-llama/Llama-3.1-70B
task=("medqa" "medmcqa" "mmlu" "pubmed")

COMMENTOUT

time python $project/src/102424_execute_inference.py \
    --inference_model meta-llama/Llama-3.1-70B \
    --task mmlu \
    --quantize_type 4bit \
    --inference_max_length 2048 \

time python $project/src/102424_execute_inference.py \
    --inference_model microsoft/Phi-3.5-mini-instruct \
    --task mmlu \
    --quantize_type none \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model epfl-llm/meditron-70b \
    --task mmlu \
    --quantize_type 4bit \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model axiong/PMC_LLaMA_13B \
    --task mmlu \
    --quantize_type none \
    --inference_max_length 2048

#############################################################

time python $project/src/102424_execute_inference.py \
    --inference_model meta-llama/Llama-3.1-70B \
    --task medmcqa \
    --quantize_type 4bit \
    --inference_max_length 2048 \

time python $project/src/102424_execute_inference.py \
    --inference_model microsoft/Phi-3.5-mini-instruct \
    --task medmcqa \
    --quantize_type none \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model epfl-llm/meditron-70b \
    --task medmcqa \
    --quantize_type 4bit \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model axiong/PMC_LLaMA_13B \
    --task medmcqa \
    --quantize_type none \
    --inference_max_length 2048

#############################################################

time python $project/src/102424_execute_inference.py \
    --inference_model meta-llama/Llama-3.1-70B \
    --task medqa \
    --quantize_type 4bit \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model microsoft/Phi-3.5-mini-instruct \
    --task medqa \
    --quantize_type none \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model epfl-llm/meditron-70b \
    --task medqa \
    --quantize_type 4bit \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model axiong/PMC_LLaMA_13B \
    --task medqa \
    --quantize_type none \
    --inference_max_length 2048

############################################################

time python $project/src/102424_execute_inference.py \
    --inference_model meta-llama/Llama-3.1-70B \
    --task pubmedqa \
    --quantize_type 4bit \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model microsoft/Phi-3.5-mini-instruct \
    --task pubmedqa \
    --quantize_type none \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model epfl-llm/meditron-70b \
    --task pubmedqa \
    --quantize_type 4bit \
    --inference_max_length 2048

time python $project/src/102424_execute_inference.py \
    --inference_model axiong/PMC_LLaMA_13B \
    --task pubmedqa \
    --quantize_type none \
    --inference_max_length 2048

#############################################################

echo "Done"
