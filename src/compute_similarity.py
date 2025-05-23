import torch
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_cosine_similarity_batches(
    data_tensor,
    batch_size=4000,
    chunk_size=4000,
    top_k=None,
    device="cuda"
):
    num_rows = data_tensor.size(0)
    similarity_dict = {}

    for i in tqdm(range(0, num_rows, batch_size), desc="Computing similarities"):
        batch = data_tensor[i:i + batch_size].to(device)

        for j in range(0, num_rows, chunk_size):
            chunk = data_tensor[j:j + chunk_size].to(device)

            sims = torch.matmul(batch, chunk.T)  

            for row_idx in range(batch.size(0)):
                src = i + row_idx
                if src >= num_rows:
                    continue

                sim_row = sims[row_idx]
                indices = torch.arange(j, j + chunk.size(0), device=device)
                if top_k:
                    top_vals, top_indices = torch.topk(sim_row, k=min(top_k + 1, sim_row.size(0)))
                    for dst, val in zip(indices[top_indices], top_vals):
                        if dst.item() != src:
                            similarity_dict.setdefault(src, []).append((dst.item(), val.item()))
                else:
                    for dst, val in zip(indices, sim_row):
                        if dst.item() != src:
                            similarity_dict.setdefault(src, []).append((dst.item(), val.item()))

            del chunk, sims
            torch.cuda.empty_cache()

        if top_k:
            # Retain only top_k globally for each row
            for src in similarity_dict:
                similarity_dict[src] = sorted(similarity_dict[src], key=lambda x: -x[1])[:top_k]

        del batch
        torch.cuda.empty_cache()

    return similarity_dict

def save_similarity(sim_dict, output_path):
    torch.save(sim_dict, output_path)
    print(f"\u2705 Similarity saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--chunk_size", type=int, default=4000)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\ud83d\ude80 Using device: {device}")

    # Load and normalize data
    df = pd.read_csv(args.input_csv)
    data = torch.tensor(df.values, dtype=torch.float32)
    data = torch.nn.functional.normalize(data, p=2, dim=1)

    # Compute similarity
    sim_dict = compute_cosine_similarity_batches(
        data,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        device=device
    )

    # Save
    save_similarity(sim_dict, args.output_path)

if __name__ == "__main__":
    main()
