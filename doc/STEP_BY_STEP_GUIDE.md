# GNN4_IO_4 Pipeline: Step-by-Step Execution Guide

This document provides detailed instructions for running the GNN4_IO_4 pipeline, which combines the TabGNN approach with IODiagnose methods for I/O performance prediction.

## Prerequisites

1. **Environment Setup**
   - Python 3.8+ with required packages installed
   - CUDA-compatible GPU (recommended for faster training)
   - Conda environment (recommended for dependency management)

2. **Required Packages**
   ```bash
   # Create and activate conda environment
   conda create -n gnn_env python=3.10
   conda activate gnn_env
   
   # Install required packages
   pip install pandas numpy torch scikit-learn matplotlib
   pip install torch-geometric
   pip install lightgbm catboost xgboost
   pip install ray[tune] pytorch-tabnet
   ```

## Pipeline Steps

### 1. Data Preparation

1. **Prepare your dataset**
   - Ensure your CSV file contains the necessary I/O counter features
   - Important features should include:
     - POSIX_SEQ_WRITES
     - POSIX_SIZE_READ_1K_10K
     - POSIX_SIZE_READ_10K_100K
     - POSIX_unique_bytes
     - POSIX_SIZE_READ_0_100
     - POSIX_SIZE_WRITE_100K_1M
     - POSIX_write_only_bytes
     - POSIX_MEM_NOT_ALIGNED
     - POSIX_FILE_NOT_ALIGNED
   - Ensure your target column (e.g., "tag") is included

2. **Create directory structure**
   ```bash
   mkdir -p data
   mkdir -p logs/training
   mkdir -p logs/tuning
   mkdir -p logs/slurm
   ```

3. **Place your dataset in the data directory**
   ```bash
   cp /path/to/your/dataset.csv data/
   ```

### 2. Basic Model Training

1. **Run a single model**
   ```bash
   # Modify the run_models.slurm file to set your data path and target column
   # Then submit the job
   sbatch slurm/run_models.slurm
   ```

   Alternatively, you can run a specific model directly:
   ```bash
   python scripts/train.py \
       --data_path data/your_dataset.csv \
       --output_dir logs/training/lightgbm \
       --target_column tag \
       --important_features POSIX_SEQ_WRITES POSIX_SIZE_READ_1K_10K POSIX_SIZE_READ_10K_100K POSIX_unique_bytes POSIX_SIZE_READ_0_100 POSIX_SIZE_WRITE_100K_1M POSIX_write_only_bytes POSIX_MEM_NOT_ALIGNED POSIX_FILE_NOT_ALIGNED \
       --similarity_threshold 0.05 \
       --model_type lightgbm
   ```

2. **Run all models for comparison**
   - Edit the MODEL_TYPE variable in run_models.slurm to "all"
   - Submit the job:
   ```bash
   sbatch slurm/run_models.slurm
   ```

3. **View results**
   - Check the output directory for model checkpoints and metrics
   - Review the comparison report to see which model performed best
   ```bash
   cat logs/training/GNN4_IO_4/comparison_report.json
   ```

### 3. Hyperparameter Tuning

1. **Run hyperparameter tuning**
   ```bash
   # Modify the run_hyperparameter_tuning.slurm file to set your data path and target column
   # Then submit the job
   sbatch slurm/run_hyperparameter_tuning.slurm
   ```

   You can also tune specific components by modifying the TUNE_COMPONENT variable:
   - "graph": Tune only graph construction parameters
   - "gnn": Tune only GNN hyperparameters
   - "tabular": Tune only tabular model hyperparameters
   - "all": Tune all components (default)

2. **Train models with best hyperparameters**
   - This is automatically done at the end of the tuning job
   - Alternatively, you can run it manually:
   ```bash
   python scripts/train_with_best_config.py \
       --data_path data/your_dataset.csv \
       --output_dir logs/tuning/GNN4_IO_4/models \
       --target_column tag \
       --config_path logs/tuning/GNN4_IO_4/best_configs.json
   ```

3. **View tuning results**
   ```bash
   cat logs/tuning/GNN4_IO_4/best_configs.json
   ```

### 4. Custom Pipeline Execution

For more control over the pipeline, you can execute each step individually:

1. **Data processing and graph construction**
   ```python
   from src.data import IODataProcessor
   
   # Define important features
   important_features = [
       "POSIX_SEQ_WRITES",
       "POSIX_SIZE_READ_1K_10K",
       "POSIX_SIZE_READ_10K_100K",
       "POSIX_unique_bytes",
       "POSIX_SIZE_READ_0_100",
       "POSIX_SIZE_WRITE_100K_1M",
       "POSIX_write_only_bytes",
       "POSIX_MEM_NOT_ALIGNED",
       "POSIX_FILE_NOT_ALIGNED"
   ]
   
   # Create data processor
   processor = IODataProcessor(
       data_path="data/your_dataset.csv",
       important_features=important_features,
       similarity_thresholds={f: 0.05 for f in important_features}
   )
   
   # Load and preprocess data
   processor.load_data()
   processor.preprocess_data()
   
   # Create PyG data
   data = processor.create_combined_pyg_data(target_column="tag")
   
   # Split data
   data = processor.train_val_test_split(data)
   
   # Save processed data
   processor.save_processed_data("data/processed", target_column="tag")
   ```

2. **Train a TabGNN model**
   ```python
   import torch
   from src.models.gnn import TabGNNRegressor
   
   # Set device
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Move data to device
   data = data.to(device)
   
   # Create model
   model = TabGNNRegressor(
       in_channels=data.x.shape[1],
       hidden_channels=64,
       gnn_out_channels=64,
       mlp_hidden_channels=[64, 32],
       num_layers=2,
       num_graph_types=1,
       model_type="gcn",
       dropout=0.1
   ).to(device)
   
   # Create optimizer
   optimizer = torch.optim.Adam(
       model.parameters(),
       lr=0.001,
       weight_decay=0.0001
   )
   
   # Train model
   # (See scripts/train.py for full training loop)
   
   # Save model
   model.save_checkpoint(
       checkpoint_dir="logs/training/custom",
       filename="tabgnn_model.pt"
   )
   ```

3. **Train a tabular model**
   ```python
   from src.models.tabular import LightGBMModel
   
   # Convert data to numpy
   X = data.x.cpu().numpy()
   y = data.y.cpu().numpy()
   
   # Get train, val, test indices
   train_idx = data.train_mask.cpu().numpy()
   val_idx = data.val_mask.cpu().numpy()
   test_idx = data.test_mask.cpu().numpy()
   
   # Split data
   X_train, y_train = X[train_idx], y[train_idx]
   X_val, y_val = X[val_idx], y[val_idx]
   X_test, y_test = X[test_idx], y[test_idx]
   
   # Create model
   model = LightGBMModel(
       random_state=42,
       n_estimators=1000,
       learning_rate=0.05
   )
   
   # Train model
   model.fit(X_train, y_train, eval_set=(X_val, y_val))
   
   # Evaluate model
   metrics = model.evaluate(X_test, y_test)
   
   # Save model
   model.save("logs/training/custom/lightgbm_model.joblib")
   ```

## Troubleshooting

1. **Missing features in dataset**
   - If your dataset doesn't contain all the important features, the pipeline will still work but will create models without graph structure
   - Check the logs for warnings about missing features

2. **CUDA out of memory**
   - Reduce batch size or model size
   - Try using a smaller dataset for testing

3. **Ray Tune errors**
   - Ensure you have enough disk space for Ray Tune's checkpoints
   - Reduce the number of samples or trials if running out of resources

## Advanced Configuration

1. **Modifying graph construction parameters**
   - Edit the similarity thresholds in the Slurm scripts or command line arguments
   - Experiment with different similarity metrics ("cosine" or "euclidean")
   - Adjust the maximum number of edges per node

2. **Customizing model architectures**
   - Modify the hidden dimensions, number of layers, and dropout rates
   - Try different GNN types (GCN or GAT)
   - Experiment with different tabular models

3. **Hyperparameter tuning configuration**
   - Adjust the search spaces in src/raytune_optimizer.py
   - Modify the number of samples and maximum epochs in the Slurm script
   - Change the resources allocated per trial based on your cluster configuration
