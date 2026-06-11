import argparse
import gzip
import os
from os.path import join
import pandas as pd
import shutil
import sys
import torch
from tqdm import tqdm
import yaml


def load_dataset(args):
    dataset_fn = "dataset" if args.exp_id is None else f"dataset_{args.exp_id}"
    dataset_fn = join(args.input_dir, dataset_fn + ".csv") 
    df = pd.read_csv(dataset_fn)
    print(f"Loaded dataset\n\t'{dataset_fn}'\nwith {len(df)} rows")
    return df
#<


# def load_splits(args):
#     splits = ["train", "val", "test"]
#     if args.exp_id is not None:
#         splits = [f"{s}_{args.exp_id}" for s in splits]
#     split_files = [f"{s}.txt.gz" for s in splits]

#     res = {}
#     for split, s in zip(splits, split_files):
#         s_fn = join(args.input_dir, s)
#         with gzip.open(s_fn, "rt") as f:
#             f_ids = [int(l.strip()) for l in f.readlines()]
#         res[ split ] = f_ids  # remove exp_id if it has been added
#     return res
# #<

def load_splits(args):
    splits = ["train", "val", "test"]
    if args.exp_id is not None:
        split_files = [f"{s}_{args.exp_id}" for s in splits]
    split_files = [f"{s}.txt.gz" for s in split_files]

    res = {}
    for split, s in zip(splits, split_files):
        s_fn = join(args.input_dir, s)
        with gzip.open(s_fn, "rt") as f:
            f_ids = [int(l.strip()) for l in f.readlines()]
        print(split)
        
        res[ split ] = f_ids  # remove exp_id if it has been added
    return res

def process_split(df, split, ids, output_dir):
    mapping = {}
    
    # select rows with 'formula_id' in ids
    df_split = df[df["formula_id"].isin(ids)]
    assert (len(df_split) == len(ids)), f"Split {split} has {len(df_split)} rows, but {len(ids)} ids"
    # separate sat and unsat
    df_split_sat = df_split[df_split["label"] == 1.0]
    df_split_unsat = df_split[df_split["label"] == 0.0]
    print(f"{split}: |sat|= {len(df_split_sat)}; |unsat|= {len(df_split_unsat)}")
    # now copy the files to the folder:
    #    - output_dir/split/sat for sat files
    #    - output_dir/split/unsat for unsat files
    for label, df_split in zip(["sat", "unsat"], [df_split_sat, df_split_unsat]):
        label_dir = join(output_dir, split if split != "val" else "valid", label)
        os.makedirs(label_dir, exist_ok=True)
        for i, (_, row) in enumerate(df_split.iterrows()):
            src = join(row["folder"], row["filename"])
            assert os.path.exists(src), f"Source file {src} does not exist"
            dst = join(label_dir, f"{i}.cnf")
            shutil.copyfile(src, dst)
            mapping[row["filename"]] = dst
            print(f"Copied {src} to {dst}")
        #< for over split
    #< for over labels
    return mapping
#< 


def main(args):
    # -- 1 --
    # read datast and splits
    df = load_dataset(args)    
    print( df.head(2) )

    # -- 2 -- load splits    
    splits = load_splits(args)
    for k , v in splits.items():
        print(f"step 2, {k}: {len(v)}")
    
    # -- 3 --
    # copy dimacs files to the appropriate sat/unsat folders
    mapping = {}
    for split, ids in splits.items():
        mapping.update( process_split(df, split, ids, args.output_dir) )
    assert len(mapping) == len(df), f"Mapping has {len(mapping)} rows, but df has {len(df)} rows"

    # -- 4 -- 
    # save the mapping from original file name -> new path
    mapping_fn = join(args.output_dir, "mapping.txt")
    with open(mapping_fn, "w") as f:
        f.write("original_file,new_file\n")
        for k, v in mapping.items():
            f.write(f"{k},{v}\n")
    print(f"Saved mapping to {mapping_fn}")

    # -- 5 -- 
    # save args to a yaml file in the output directory, with pyyaml
    args.input_dir = os.path.abspath(args.input_dir)
    args.date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    args_fn = join(args.output_dir, "args.yaml")
    with open(args_fn, "w") as f:
        yaml.dump(vars(args), f)
    print(f"Saved args to {args_fn}")
#<

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G4SATBench data import")
    parser.add_argument("--input_dir", type=str, default="../output/exps/dev")
    parser.add_argument("--output_dir", type=str, default="~/workspace/G4SATBench/exported-dataset")
    # parser.add_argument("--rename", type=str, default="yes", choices=["yes", "no"], help="Rename files in the input directory")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID")
    args = parser.parse_args()
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist")
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} does not exist")
        sys.exit(1)
    
    main(args)
    print("all done.")
#<