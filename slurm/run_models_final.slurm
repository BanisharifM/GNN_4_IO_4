#!/bin/bash
#SBATCH --job-name=GNN4_IO_4
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

module load cuda
source activate gnn4_env

CONFIG_FILE="configs/experiment3.yml"
EXPERIMENT_NAME="Experiment3"
OUTPUT_DIR="logs/training/all/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

python scripts/train.py --config_file "$CONFIG_FILE" --output_dir "$OUTPUT_DIR" --device cuda
