#!/bin/bash
#SBATCH --job-name=compute_similarity
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/compute_similarity_%j.out
#SBATCH --error=logs/slurm/compute_similarity_%j.err

module load cuda
source activate gnn4_env

CSV_PATH="data/sample_train_total.csv"
OUTPUT_PATH="data/similarity_output.pt"
BATCH_SIZE=10408
CHUNK_SIZE=10408
TOP_K= # set to value like 30 or leave empty

python src/compute_similarity.py \
  --input_csv "$CSV_PATH" \
  --output_path "$OUTPUT_PATH" \
  --batch_size $BATCH_SIZE \
  --chunk_size $CHUNK_SIZE \
  ${TOP_K:+--top_k $TOP_K}
