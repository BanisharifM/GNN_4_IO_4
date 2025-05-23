import torch
import argparse
import os
from glob import glob

def merge_similarity_dicts(input_dir, output_path):
    merged = {}

    pt_files = sorted(glob(os.path.join(input_dir, "similarity_output_total.pt.rank*.pt")))
    print(f"Found {len(pt_files)} partial similarity files.")

    for pt_file in pt_files:
        partial = torch.load(pt_file)
        merged.update(partial)  # Assumes no key overlap
        print(f"Merged {pt_file} with {len(partial)} entries.")

    torch.save(merged, output_path)
    print(f"Final merged similarity saved to: {output_path}")
    print(f"Total rows: {len(merged)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    merge_similarity_dicts(args.input_dir, args.output_path)


# python merge_similarity_dicts.py \
#   --input_dir data/ \
#   --output_path data/similarity_output_merged_100K.pt
