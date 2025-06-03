#!/bin/bash
#SBATCH --job-name=merge_simi_final_1M
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/merge_simni_final_%j.out
#SBATCH --error=logs/slurm/merge_simi_final_%j.err

module load cuda
source activate gnn4_env

INPUT_DIR="/projects/bdau/mbanisharifdehkordi/gnn_data/GNN_4_IO_4/data/1M/merged_batches/"
FINAL_OUTPUT="/projects/bdau/mbanisharifdehkordi/gnn_data/GNN_4_IO_4/data/1M/merged_similarity_final_1M_ONE.pt"

srun python scripts/merge_similarity_dicts_final_1M.py \
  --input_dir "$INPUT_DIR" \
  --output_path "$FINAL_OUTPUT"
