"""
Model comparison script for GNN4_IO_4.

This script compares the performance of different models trained on I/O performance data,
including traditional tabular models and TabGNN models.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare model performance for I/O performance prediction")
    
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing model results")
    parser.add_argument("--output_file", type=str, default="comparison_report.json", help="Output file for comparison report")
    parser.add_argument("--plot_file", type=str, default="model_comparison.png", help="Output file for comparison plot")
    
    return parser.parse_args()

def load_metrics(results_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load metrics from model results directories.
    
    Args:
        results_dir (str): Directory containing model results
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping model names to metrics
    """
    metrics = {}
    
    # Get subdirectories (one for each model)
    # model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    model_dirs = glob.glob(os.path.join(results_dir, "*", "Experiment1"))
 
    for model_dir in model_dirs:
        metrics_file = os.path.join(results_dir, model_dir, "metrics.json")
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                model_metrics = json.load(f)
                metrics[model_dir] = model_metrics
                logger.info(f"Loaded metrics for {model_dir}: {model_metrics}")
        else:
            logger.warning(f"No metrics file found for {model_dir}")
    
    return metrics

def create_comparison_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create comparison table from metrics.
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary mapping model names to metrics
        
    Returns:
        pd.DataFrame: Comparison table
    """
    # Create DataFrame
    comparison = pd.DataFrame(index=metrics.keys())
    
    # Add metrics as columns
    for model, model_metrics in metrics.items():
        for metric, value in model_metrics.items():
            comparison.loc[model, metric] = value
    
    # Sort by RMSE (lower is better)
    if 'rmse' in comparison.columns:
        comparison = comparison.sort_values('rmse')
    
    return comparison

def plot_comparison(metrics: Dict[str, Dict[str, float]], output_file: str):
    """
    Plot model comparison.
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary mapping model names to metrics
        output_file (str): Output file for plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract model names and metrics
    models = list(metrics.keys())
    
    # Check which metrics are available
    available_metrics = set()
    for model_metrics in metrics.values():
        available_metrics.update(model_metrics.keys())
    
    # Select metrics to plot
    plot_metrics = ['rmse', 'mae', 'r2'] if all(m in available_metrics for m in ['rmse', 'mae', 'r2']) else list(available_metrics)
    
    # Extract metric values
    metric_values = {}
    for metric in plot_metrics:
        metric_values[metric] = [metrics[model].get(metric, 0) for model in models]
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.2
    n_metrics = len(plot_metrics)
    offsets = np.linspace(-(n_metrics-1)*width/2, (n_metrics-1)*width/2, n_metrics)
    
    # Plot bars
    for i, metric in enumerate(plot_metrics):
        ax.bar(x + offsets[i], metric_values[metric], width, label=metric.upper())
    
    # Add labels and legend
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Comparison plot saved to {output_file}")

def create_comparison_report(metrics: Dict[str, Dict[str, float]], output_file: str):
    """
    Create comparison report.
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary mapping model names to metrics
        output_file (str): Output file for report
    """
    # Create comparison table
    comparison = create_comparison_table(metrics)
    
    # Calculate improvement over baseline
    if 'lightgbm' in comparison.index and 'rmse' in comparison.columns:
        baseline_rmse = comparison.loc['lightgbm', 'rmse']
        comparison['improvement'] = (baseline_rmse - comparison['rmse']) / baseline_rmse * 100
    
    # Convert to dictionary for JSON serialization
    report = {
        'metrics': metrics,
        'comparison': comparison.to_dict(),
        'best_model': comparison.index[0] if not comparison.empty else None,
        'best_metrics': comparison.iloc[0].to_dict() if not comparison.empty else None
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Comparison report saved to {output_file}")
    
    return report

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load metrics
    metrics = load_metrics(args.results_dir)
    
    if not metrics:
        logger.error("No metrics found in results directory")
        return
    
    # Create comparison report
    report = create_comparison_report(metrics, args.output_file)
    
    # Plot comparison
    plot_comparison(metrics, args.plot_file)
    
    # Print summary
    logger.info(f"Best model: {report['best_model']}")
    logger.info(f"Best metrics: {report['best_metrics']}")

if __name__ == "__main__":
    main()
