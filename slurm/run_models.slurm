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

# === Fallback static log paths (used only if dynamic logging fails) ===
#SBATCH --output=logs/slurm/default.out
#SBATCH --error=logs/slurm/default.err

# === Experiment & model setup ===
EXPERIMENT_NAME="Experiment3"
MODEL_TYPE="all"  # Options: lightgbm, catboost, xgboost, mlp, tabnet, tabgnn, combined, all
CONDA_ENV_PATH="$HOME/.conda/envs/gnn4_env/bin/python"
TARGET_COLUMN="tag"
DATA_PATH="data/sample_train_total.csv"
OUTPUT_DIR="logs/training/${MODEL_TYPE}/${EXPERIMENT_NAME}"

# === Setup environment ===
module load cuda
source activate gnn4_env

# === Dynamic logging (SLURM variables not available at SBATCH time) ===
LOG_DIR="logs/slurm/${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/training_${SLURM_JOB_ID}.out") 2> >(tee -a "${LOG_DIR}/training_${SLURM_JOB_ID}.err" >&2)

# === Create model output directory ===
mkdir -p "$OUTPUT_DIR"

# === Important features list ===
IMPORTANT_FEATURES=(
    "POSIX_SEQ_WRITES"
    "POSIX_SIZE_READ_1K_10K"
    "POSIX_SIZE_READ_10K_100K"
    "POSIX_SIZE_READ_0_100"
    "POSIX_SIZE_WRITE_100K_1M"
    "POSIX_MEM_NOT_ALIGNED"
    "POSIX_FILE_NOT_ALIGNED"
    "POSIX_SIZE_WRITE_0_100"
    "POSIX_SIZE_WRITE_10K_100K"
)

FEATURES_STR=$(IFS=" "; echo "${IMPORTANT_FEATURES[*]}")

# === Model runner function ===
run_model() {
    local model=$1
    local model_dir="logs/training/${model}/${EXPERIMENT_NAME}"
    
    echo "Running $model model..."
    mkdir -p "$model_dir"

    $CONDA_ENV_PATH scripts/train.py \
        --data_path "$DATA_PATH" \
        --output_dir "$model_dir" \
        --target_column "$TARGET_COLUMN" \
        --important_features "$FEATURES_STR" \
        --similarity_threshold 0.05 \
        --model_type "$model" \
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

    echo "$model model completed."
}

# === Main execution ===
echo "Starting GNN4_IO_4 pipeline at $(date)"

if [ "$MODEL_TYPE" = "all" ]; then
    for model in lightgbm catboost xgboost mlp tabnet tabgnn combined; do
        run_model "$model"
    done

    $CONDA_ENV_PATH scripts/compare_models.py \
        --results_dir "logs/training" \
        --output_file "logs/training/comparison_report_${EXPERIMENT_NAME}.json" \
        --plot_file "logs/training/model_comparison_${EXPERIMENT_NAME}.png"
else
    run_model "$MODEL_TYPE"
fi

echo "All jobs completed at $(date)"



# IMPORTANT_FEATURES=(
#     "POSIX_SEQ_WRITES"
#     "POSIX_SIZE_READ_1K_10K"
#     "POSIX_SIZE_READ_10K_100K"
#     "POSIX_SIZE_READ_0_100"
#     "POSIX_SIZE_WRITE_100K_1M"
#     "POSIX_write_only_bytes"
#     "POSIX_MEM_NOT_ALIGNED"
#     "POSIX_FILE_NOT_ALIGNED"
#     "POSIX_unique_bytes"
#     "POSIX_SEEKS"
#     "POSIX_OPENS"
# )
