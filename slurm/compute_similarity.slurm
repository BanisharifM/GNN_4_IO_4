#!/bin/bash
#SBATCH --job-name=compute_similarity_6M
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:8                
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/compute_similarity_%j.out
#SBATCH --error=logs/slurm/compute_similarity_%j.err

# === Load environment ===
module load cuda
source activate gnn4_env

# === Set distributed environment variables ===
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + RANDOM % 20000)) 
export WORLD_SIZE=4  

CSV_PATH="data/sample_6M.csv"
OUTPUT_PATH="data/similarity_output_6M.pt"
BATCH_SIZE=10408
CHUNK_SIZE=10408
TOP_K=30

srun python src/compute_similarity.py \
  --input_csv "$CSV_PATH" \
  --output_path "$OUTPUT_PATH" \
  --batch_size $BATCH_SIZE \
  --chunk_size $CHUNK_SIZE \
  --top_k $TOP_K \
  --world_size $WORLD_SIZE
