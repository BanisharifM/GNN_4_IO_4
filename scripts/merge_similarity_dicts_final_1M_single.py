import torch
import os
import argparse
from glob import glob
from tqdm import tqdm


def merge_distributed_row_batches_single(input_dir_prefix, world_size, output_path):
    merged_dict = {}
    total_entries = 0

    for rank in range(world_size):
        rank_dir = f"{input_dir_prefix}_rows_rank{rank}"
        if not os.path.isdir(rank_dir):
            print(f"⚠️ Rank directory not found: {rank_dir}")
            continue

        pt_files = sorted(glob(os.path.join(rank_dir, "*.pt")))
        print(f"📂 [Rank {rank}] Found {len(pt_files)} batch files in {rank_dir}")

        for pt_file in tqdm(pt_files, desc=f"Merging rank {rank}"):
            try:
                partial = torch.load(pt_file)
                merged_dict.update(partial)
                total_entries += len(partial)
                del partial
            except Exception as e:
                print(f"❌ Failed to load {pt_file}: {e}")

    torch.save(merged_dict, output_path)
    print(f"✅ Saved final merged file to {output_path} with {total_entries} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_prefix", type=str, required=True,
                        help="Prefix of input folders like 'similarity_output' (expects _rows_rank0, _rows_rank1, ...)")
    parser.add_argument("--world_size", type=int, required=True,
                        help="Total number of ranks used in the job")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to store the final single merged .pt file")
    args = parser.parse_args()

    merge_distributed_row_batches_single(args.input_dir_prefix, args.world_size, args.output_path)
