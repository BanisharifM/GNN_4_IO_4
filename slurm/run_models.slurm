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
#SBATCH --output=logs/slurm/GNN4_IO_4_%j.out
#SBATCH --error=logs/slurm/GNN4_IO_4_%j.err

# Load required modules
module load cuda

# Activate conda environment (replace with your environment name)
source activate gnn_env

# Create logs directory if it doesn't exist
mkdir -p logs/slurm
mkdir -p logs/training

# Set variables for the workflow
DATA_PATH="data/sample_train_100.csv"   #/u/mbanisharifdehkordi/Github/GNN_4_IO_3/data/sample_train_total.csv
OUTPUT_DIR="logs/training/GNN4_IO_4"
TARGET_COLUMN="tag"
MODEL_TYPE="all"  # Options: lightgbm, catboost, xgboost, mlp, tabnet, tabgnn, combined, all

# Define important features for graph construction
IMPORTANT_FEATURES=(
    "POSIX_SEQ_WRITES"
    "POSIX_SIZE_READ_1K_10K"
    "POSIX_SIZE_READ_10K_100K"
    "POSIX_unique_bytes"
    "POSIX_SIZE_READ_0_100"
    "POSIX_SIZE_WRITE_100K_1M"
    "POSIX_write_only_bytes"
    "POSIX_MEM_NOT_ALIGNED"
    "POSIX_FILE_NOT_ALIGNED"
)

# Convert array to space-separated string
FEATURES_STR=$(IFS=" "; echo "${IMPORTANT_FEATURES[*]}")

# Function to run a specific model
run_model() {
    local model_type=$1
    local model_output_dir="$OUTPUT_DIR/$model_type"
    
    echo "Running $model_type model..."
    mkdir -p $model_output_dir
    
    python scripts/train.py \
        --data_path $DATA_PATH \
        --output_dir $model_output_dir \
        --target_column $TARGET_COLUMN \
        --important_features $FEATURES_STR \
        --similarity_threshold 0.05 \
        --model_type $model_type \
        --gnn_type gcn \
        --hidden_dim 64 \
        --num_layers 2 \
        --dropout 0.1 \
        --batch_size 32 \
        --epochs 100 \
        --lr 0.001 \
        --weight_decay 0.0001 \
        --patience 10 \
        --seed 42 \
        --device cuda
    
    echo "$model_type model completed."
}

echo "Starting GNN4_IO_4 pipeline at $(date)"

# Run models based on the specified model type
if [ "$MODEL_TYPE" = "all" ]; then
    # Run all models
    for model in lightgbm catboost xgboost mlp tabnet tabgnn combined; do
        run_model $model
    done
    
    # Create comparison report
    python scripts/compare_models.py \
        --results_dir $OUTPUT_DIR \
        --output_file $OUTPUT_DIR/comparison_report.json \
        --plot_file $OUTPUT_DIR/model_comparison.png
else
    # Run specific model
    run_model $MODEL_TYPE
fi

echo "All jobs completed at $(date)"
