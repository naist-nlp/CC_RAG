#!/bin/bash
#SBATCH -p lang_mem_long
#SBATCH -c 2
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=evaluate
#SBATCH -o logs/slurm-%x-%j.log

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

<< COMMENTOUT
db_path: pubmed, statpearls, textbooks, wikipedia
rag_model: ncbi/MedCPT-Query-Encoder, allenai/specter, facebook/contriever
models: microsoft/Phi-3.5-mini-instruct, axiong/PMC_LLaMA_13B, meta-llama/Llama-3.1-70B, epfl-llm/meditron-70b  meta-llama/Llama-2-70b-chat-hf
COMMENTOUT

###############ここ変更##################
base_model_name=microsoft/Phi-3.5-mini-instruct
rag_db=textbooks
##################################

task=medqa
rag_model_name=facebook/contriever
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

rag_model_name=allenai/specter
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

rag_model_name=ncbi/MedCPT-Query-Encoder
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

task=mmlu
rag_model_name=facebook/contriever
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

rag_model_name=allenai/specter
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

rag_model_name=ncbi/MedCPT-Query-Encoder
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

task=pubmedqa
rag_model_name=facebook/contriever
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

rag_model_name=allenai/specter
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

rag_model_name=ncbi/MedCPT-Query-Encoder
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

task=medmcqa
rag_model_name=facebook/contriever
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

rag_model_name=allenai/specter
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

rag_model_name=ncbi/MedCPT-Query-Encoder
max_length=512
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task


max_length=256
top_k=3
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task

top_k=1
echo "RAG DB: $rag_db, RAG MODEL: $rag_model_name, TASK: $task, BASE MODEL: $base_model_name, MAX LENGTH: $max_length, TOP K: $top_k"
time python $project/src/102624_evaluate_results_with_rag.py \
    --base_model_name $base_model_name \
    --rag_model_name $rag_model_name \
    --rag_db $rag_db \
    --top_k $top_k \
    --max_length $max_length \
    --task $task
