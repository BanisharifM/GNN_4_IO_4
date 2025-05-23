import torch
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from torch.multiprocessing import Process

def setup(rank, world_size):
    if 'MASTER_ADDR' not in os.environ or 'MASTER_PORT' not in os.environ:
        raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set in the environment by SLURM script.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count()) 

def cleanup():
    dist.destroy_process_group()

def compute_cosine_similarity_distributed(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    df = pd.read_csv(args.input_csv)
    data = torch.tensor(df.values, dtype=torch.float32)
    data = torch.nn.functional.normalize(data, p=2, dim=1)
    data = data.to(device)

    num_rows = data.size(0)
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    top_k = args.top_k
    similarity_dict = {}

    # Split the data for this rank
    local_indices = list(range(rank, num_rows, world_size))

    for i in tqdm(local_indices, desc=f"Rank {rank} computing", position=rank):
        row = data[i].unsqueeze(0)  # 1 x D
        similarities = []
        for j in range(0, num_rows, chunk_size):
            chunk = data[j:j+chunk_size]
            sims = torch.matmul(row, chunk.T).squeeze(0)  # 1 x chunk -> chunk
            dst_indices = torch.arange(j, j + chunk.size(0), device=device)
            if top_k:
                vals, indices = torch.topk(sims, min(top_k + 1, sims.size(0)))
                filtered = [(int(dst_indices[idx]), float(vals[k]))
                            for k, idx in enumerate(indices) if dst_indices[idx] != i]
                similarities.extend(filtered[:top_k])
            else:
                similarities.extend([(int(dst_indices[k]), float(sims[k]))
                                     for k in range(sims.size(0)) if dst_indices[k] != i])

        similarity_dict[i] = similarities

    # Save partial result
    partial_file = f"{args.output_path}.rank{rank}.pt"
    torch.save(similarity_dict, partial_file)
    print(f"[Rank {rank}] Saved partial result to {partial_file}")
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--chunk_size", type=int, default=4000)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()

    compute_cosine_similarity_distributed(args.rank, args.world_size, args)

if __name__ == "__main__":
    main()