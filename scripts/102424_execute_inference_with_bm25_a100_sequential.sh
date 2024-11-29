#!/bin/bash
#SBATCH -p lang_gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=lang
#SBATCH --job-name=inference-with-bm25-a100
#SBATCH -o logs/slurm-%x-%j.log

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
MODELS=("axiong/PMC_LLaMA_13B", microsoft/Phi-3.5-mini-instruct meta-llama/Llama-2-70b-chat-hf
TASKS=("medqa" "medmcqa" "mmlu" "pubmedqa")
RAGDB=("statpearls" "texatbooks" "pubmed" "wikipedia")
COMMENTOUT

inference_model=epfl-llm/meditron-70b
rag_db=textbooks
inference_max_length=2048
quantize_type=4bit

task=medmcqa
rag_max_length=512
top_k=3
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

top_k=1
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

rag_max_length=256
top_k=3
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

top_k=1
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

task=medqa
rag_max_length=512
top_k=3
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

top_k=1
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

rag_max_length=256
top_k=3
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

top_k=1
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

task=pubmedqa
rag_max_length=512
top_k=3
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

top_k=1
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

rag_max_length=256
top_k=3
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

top_k=1
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

task=mmlu
rag_max_length=512
top_k=3
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

top_k=1
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

rag_max_length=256
top_k=3
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type

top_k=1
time python $project/src/102424_execute_inference_bm25.py \
    --inference_model $inference_model \
    --rag_db $rag_db \
    --task $task \
    --top_k $top_k \
    --rag_max_length $rag_max_length \
    --inference_max_length $inference_max_length \
    --quantize_type $quantize_type
