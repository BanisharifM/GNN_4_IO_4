#!/bin/bash
#SBATCH --job-name=compute_similarity
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/compute_similarity_%j.out
#SBATCH --error=logs/slurm/compute_similarity_%j.err

module load cuda
source activate gnn4_env

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$((12000 + RANDOM % 1000))
export WORLD_SIZE=$SLURM_NTASKS

CSV_PATH="data/sample_train_total_normalized.csv"
OUTPUT_PATH="data/similarity_output.pt"
BATCH_SIZE=10408
CHUNK_SIZE=10408
TOP_K=30

srun python src/compute_similarity.py \
  --input_csv "$CSV_PATH" \
  --output_path "$OUTPUT_PATH" \
  --batch_size $BATCH_SIZE \
  --chunk_size $CHUNK_SIZE \
  --top_k $TOP_K \
  --world_size $WORLD_SIZE \
  --rank $SLURM_PROCID
