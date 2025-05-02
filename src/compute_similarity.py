# compute_similarity.py
import torch
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_cosine_similarity_batches(data_tensor, batch_size=4000, top_k=None, device='cuda'):
    num_rows = data_tensor.size(0)
    similarity_dict = {}

    for i in tqdm(range(0, num_rows, batch_size), desc="Computing similarities"):
        batch = data_tensor[i:i + batch_size].to(device)
        sims = torch.matmul(batch, data_tensor.T.to(device))  # Cosine because data is normalized

        if top_k:
            top_vals, top_indices = torch.topk(sims, k=top_k + 1, dim=1)  # +1 to skip self
            for row_idx, (vals, indices) in enumerate(zip(top_vals, top_indices)):
                src = i + row_idx
                filtered = [(int(dst), float(sim)) for dst, sim in zip(indices, vals) if dst != src]
                similarity_dict[src] = filtered[:top_k]
        else:
            # Keep all similarities (can be huge!)
            similarity_dict.update({
                i + row_idx: [(int(j), float(sims[row_idx, j])) for j in range(num_rows) if j != (i + row_idx)]
                for row_idx in range(batch.size(0))
            })

        del batch, sims
        torch.cuda.empty_cache()

    return similarity_dict

def save_similarity(sim_dict, output_path):
    torch.save(sim_dict, output_path)
    print(f"✅ Similarity saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load and normalize data
    df = pd.read_csv(args.input_csv)
    data = torch.tensor(df.values, dtype=torch.float32).to(device)
    data = torch.nn.functional.normalize(data, p=2, dim=1)

    # Compute similarity
    sim_dict = compute_cosine_similarity_batches(data, batch_size=args.batch_size, top_k=args.top_k, device=device)

    # Save
    save_similarity(sim_dict, args.output_path)

if __name__ == "__main__":
    main()
