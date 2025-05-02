#!/bin/bash
#SBATCH --job-name=compute_similarity
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/compute_similarity_%j.out
#SBATCH --error=logs/slurm/compute_similarity_%j.err

CONDA_ENV_PATH="$HOME/.conda/envs/gnn4_env/bin/python"
module load cuda
# source activate gnn4_env

# Arguments
CSV_PATH="data/sample_train_100.csv"
OUTPUT_PATH="data/similarity_output.pt"
BATCH_SIZE=4000
TOP_K=  # Leave empty or set value like 30

# Run
 $CONDA_ENV_PATH src/compute_similarity.py \
    --input_csv "$CSV_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    ${TOP_K:+--top_k $TOP_K}
