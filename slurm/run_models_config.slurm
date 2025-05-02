#!/bin/bash
#SBATCH --job-name=GNN4_IO_4
#SBATCH --account=bdau-delta-gpu    
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# === Setup environment ===
module load cuda
source activate gnn4_env

# === Read config file ===
CONFIG_FILE="configs/experiment1.yml"
EXPERIMENT_NAME=$(yq eval '.experiment_name' "$CONFIG_FILE")
MODEL_TYPE=$(yq eval '.model_type' "$CONFIG_FILE")
DATA_PATH=$(yq eval '.data_path' "$CONFIG_FILE")
TARGET_COLUMN=$(yq eval '.target_column' "$CONFIG_FILE")
IMPORTANT_FEATURES=$(yq eval '.important_features | join(" ")' "$CONFIG_FILE")
OUTPUT_DIR="logs/training/${MODEL_TYPE}/${EXPERIMENT_NAME}"

HIDDEN_DIM=$(yq eval '.hyperparams.hidden_dim' "$CONFIG_FILE")
NUM_LAYERS=$(yq eval '.hyperparams.num_layers' "$CONFIG_FILE")
DROPOUT=$(yq eval '.hyperparams.dropout' "$CONFIG_FILE")
BATCH_SIZE=$(yq eval '.hyperparams.batch_size' "$CONFIG_FILE")
EPOCHS=$(yq eval '.hyperparams.epochs' "$CONFIG_FILE")
LR=$(yq eval '.hyperparams.lr' "$CONFIG_FILE")
WEIGHT_DECAY=$(yq eval '.hyperparams.weight_decay' "$CONFIG_FILE")
PATIENCE=$(yq eval '.hyperparams.patience' "$CONFIG_FILE")
SEED=$(yq eval '.hyperparams.seed' "$CONFIG_FILE")
SIM_THRESHOLD=$(yq eval '.similarity_threshold' "$CONFIG_FILE")

mkdir -p "$OUTPUT_DIR"

# === Model runner ===
run_model() {
    local model=$1
    local model_dir="logs/training/${model}/${EXPERIMENT_NAME}"
    mkdir -p "$model_dir"

    echo "Running $model model..."
    python scripts/train.py \
        --data_path "$DATA_PATH" \
        --output_dir "$model_dir" \
        --target_column "$TARGET_COLUMN" \
        --important_features "$IMPORTANT_FEATURES" \
        --similarity_threshold "$SIM_THRESHOLD" \
        --model_type "$model" \
        --gnn_type gcn \
        --hidden_dim "$HIDDEN_DIM" \
        --num_layers "$NUM_LAYERS" \
        --dropout "$DROPOUT" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --patience "$PATIENCE" \
        --seed "$SEED" \
        --device cuda

    echo "$model model completed."
}

# === Main execution ===
echo "Starting Experiment: $EXPERIMENT_NAME"

if [ "$MODEL_TYPE" = "all" ]; then
    for model in lightgbm catboost xgboost mlp tabnet tabgnn combined; do
        run_model "$model"
    done
    python scripts/compare_models.py \
        --results_dir "logs/training" \
        --output_file "logs/training/comparison_report_${EXPERIMENT_NAME}.json" \
        --plot_file "logs/training/model_comparison_${EXPERIMENT_NAME}.png"
else
    run_model "$MODEL_TYPE"
fi

echo "All jobs completed at $(date)"
