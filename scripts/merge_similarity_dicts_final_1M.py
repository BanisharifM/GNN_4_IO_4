import os
import torch
import argparse
from glob import glob
from tqdm import tqdm


def merge_all_batches(input_dir, output_path):
    merged_dict = {}
    pt_files = sorted(glob(os.path.join(input_dir, "merged_*.pt")))
    print(f"Found {len(pt_files)} .pt files to merge in {input_dir}")

    total_entries = 0
    for pt_file in tqdm(pt_files, desc="Merging final batches"):
        try:
            partial_dict = torch.load(pt_file)
            merged_dict.update(partial_dict)
            total_entries += len(partial_dict)
        except Exception as e:
            print(f"❌ Failed to load {pt_file}: {e}")

    torch.save(merged_dict, output_path)
    print(f"✅ Final merged file saved to {output_path} with {total_entries} total entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing merged_*.pt batch files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the final merged .pt file")
    args = parser.parse_args()

    merge_all_batches(args.input_dir, args.output_path)
