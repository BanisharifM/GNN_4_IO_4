import torch
import joblib
import pandas as pd
import numpy as np
import os
import sys
import yaml
import shap
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import IODataProcessor
from src.models.gnn import TabGNNRegressor
from src.models.tabular import LightGBMModel, TabGNNTabularModel

# === Load YAML config ===
with open("configs/experiment6.yml", "r") as f:
    config = yaml.safe_load(f)

# === Extract config values ===
csv_path = config["data_path"]
target_column = config["target_column"]
precomputed_path = config["precomputed_similarity_path"]
similarity_threshold = config["similarity_threshold"]
hidden_dim = config["hidden_dim"]
num_layers = config["num_layers"]
dropout = config["dropout"]
seed = config["seed"]

output_dir = "logs/training/all/Experiment5/combined"
tabular_model_path = os.path.join(output_dir, "tabular_part.joblib")
gnn_model_path = os.path.join(output_dir, "tabgnn_part.pt")

# === Load and preprocess data ===
data_processor = IODataProcessor(
    data_path=csv_path,
    important_features=None,
    similarity_thresholds=None,
    precomputed_similarity_path=precomputed_path
)
data_processor.load_data()
data_processor.preprocess_data()
data = data_processor.create_combined_pyg_data(target_column=target_column)
data = data_processor.train_val_test_split(data, random_state=seed)

# === Reconstruct and load TabGNN model ===
gnn_model = TabGNNRegressor(
    in_channels=data.x.shape[1],
    hidden_channels=hidden_dim,
    gnn_out_channels=hidden_dim,
    mlp_hidden_channels=[hidden_dim, hidden_dim // 2],
    num_layers=num_layers,
    num_graph_types=1,
    model_type="gcn",  # Assumed default
    dropout=dropout
)
checkpoint = torch.load(gnn_model_path, map_location="cpu")
gnn_model.load_state_dict(checkpoint["model_state_dict"])
gnn_model.eval()

# === Load tabular model ===
tabular_checkpoint = joblib.load(tabular_model_path)
tabular_model = tabular_checkpoint["model"]
tabular_model.model_type = "lightgbm"

# === Wrap in combined model ===
combined_model = TabGNNTabularModel(
    gnn_model=gnn_model,
    tabular_model=tabular_model,
    use_original_features=True
)

# === Predict on test set ===
with torch.no_grad():
    output = combined_model.predict(
        x=data.x,
        edge_indices=[data.edge_index],
        batch=None
    )

    test_mask = data.test_mask
    y_pred = output[test_mask].squeeze() if isinstance(output, torch.Tensor) else output[test_mask]
    y_true = data.y[test_mask].cpu().numpy()
    X_test = data.x[test_mask].cpu().numpy()

# === Metrics ===
df_results = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred,
    'error': y_pred - y_true
})

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(df_results.head())
print(f"\nRMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

# === SHAP Analysis ===
print("\nPreparing SHAP input using combined GNN + tabular features...")

# Step 1: construct raw + GNN features
X_raw = data.x.cpu().numpy()[test_mask]
X_gnn = combined_model.extract_embeddings(
    x=data.x,
    edge_indices=[data.edge_index],
    batch=None
)[test_mask]
X_test_combined = combined_model.combine_features(X_raw, X_gnn)

# Step 2: run SHAP
explainer = shap.Explainer(tabular_model)
shap_values = explainer(X_test_combined)

# Step 3: build feature names
raw_features = data_processor.data.columns.tolist()
raw_features.remove(target_column)
embedding_dim = X_gnn.shape[1]
embedding_features = [f"gnn_emb_{i}" for i in range(embedding_dim)]
combined_feature_names = raw_features + embedding_features

shap_df = pd.DataFrame(shap_values.values, columns=combined_feature_names)

# Step 4: save results
full_results = pd.concat([df_results.reset_index(drop=True), shap_df], axis=1)
full_results.to_csv(os.path.join(output_dir, "predictions_with_shap.csv"), index=False)

print("\nTop SHAP feature importances (absolute mean):")
print(np.abs(shap_df).mean().sort_values(ascending=False).head(10))

# Optional plot
shap.plots.beeswarm(shap_values, max_display=15)
