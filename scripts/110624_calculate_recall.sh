#!/bin/bash
#SBATCH -p lang_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=plot-calibration-curve
#SBATCH -o logs/slurm-%x-%j.log
set -eu

project=/cl/home2/shintaro/rag-notebook/shintaro
source $project/.venv/bin/activate

python $project/src/110624_calculate_recall.py


echo "Job finished"
