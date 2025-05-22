#!/bin/bash
#SBATCH --job-name=compute_similarity
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/compute_similarity_%j.out
#SBATCH --error=logs/slurm/compute_similarity_%j.err

CONDA_ENV_PATH="$HOME/.conda/envs/gnn4_env/bin/python"
module load cuda
# source activate gnn4_env

# Arguments
CSV_PATH="data/sample_train_total.csv"
OUTPUT_PATH="data/similarity_output.pt"
BATCH_SIZE=2000
TOP_K=  # Leave empty or set value like 30

# Run
 $CONDA_ENV_PATH src/compute_similarity.py \
    --input_csv "$CSV_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    ${TOP_K:+--top_k $TOP_K}
