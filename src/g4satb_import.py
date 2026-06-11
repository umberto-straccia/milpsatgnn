import argparse
import gzip
import os
from os.path import join
import pandas as pd
import sys
import torch
from tqdm import tqdm

from data.data_generation import build_dimacs_representation

def rename_files(input_dir):  # , append_sat="no"):
    counter = 0
    for split in ['train', 'test', 'valid']:
        for label in ['sat', 'unsat']:
            dir_path = join(input_dir, split, label)
            files = [f for f in os.listdir(dir_path) if f.endswith('.cnf')]
            for file in files:
                old_file = os.path.join(dir_path, file)
                new_file = os.path.join(dir_path, f"{counter}_{label}.cnf")
                os.rename(old_file, new_file)
                print(f"Renamed {old_file} to {new_file}")
                counter += 1
    print(f"Renamed {counter} files in total.")
#<


def collect_files(input_dir):
    data = {}
    total_files = 0
    for split in ['train', 'test', 'valid']:
        data[split] = {}
        for label in ['sat', 'unsat']:
            dir_path = join(input_dir, split, label)
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.cnf')]
            total_files += len(files)
            data[split][label] = sorted(files)
    print(f"Total files: {total_files}")
    return data
#<


def process_split(input_dir, split, sat_files, unsat_files):
    res = []
    label = 1.0
    for file in tqdm(sat_files):
        filename = os.path.basename(file)
        formula_id = int( filename.split(".")[0].split("_")[0])  # filename is e.g. 1_sat.cnf 
        res.append({"formula_id": formula_id, "label": label, "filename": filename, "folder": join(input_dir, split, "sat")})
    label = 0.0
    for file in tqdm(unsat_files):
        filename = os.path.basename(file)
        formula_id = int( filename.split(".")[0].split("_")[0])  # filename is e.g. 1_sat.cnf 
        res.append({"formula_id": formula_id, "label": label, "filename": filename, "folder": join(input_dir, split, "unsat")})
    df = pd.DataFrame.from_records(res)
    print(df.groupby("label").size().to_frame(name="count").reset_index())
    return df
#<


def save_splits(file_dict, output_dir):
    for split, labels in file_dict.items():
        filenames = []
        for _, files in labels.items():
            filenames.extend([os.path.basename(fn) for fn in files])

        formula_ids = [int( fn.split(".")[0].split("_")[0] ) for fn in filenames] 
        
        out_txt = "\n".join([str(x) for x in formula_ids])
        name = "val" if split == "valid" else split
        out_fn = os.path.join(output_dir, f"{name}.txt.gz")
        with gzip.open(out_fn, "wt") as f:
            f.write(out_txt)
        print(f"Saved {split} split to {out_fn}")
#<


def prepare_dataframe(file_dict):
    dfs = {}
    for split, labels in file_dict.items():
        df =  process_split(args.input_dir, split, labels["sat"], labels["unsat"])
        dfs[split] = df
    df = pd.concat(dfs.values(), ignore_index=True)
    print(df.groupby("label").size().to_frame(name="count").reset_index())
    df = df.set_index("formula_id")
    # add columns - meaningless in case of import
    df["seed"] = 1
    df["k"] = 3
    df["n_var"] = 0
    df["n_clauses"] = 0
    df["var_values"] = ""
    df["time"] = 0
    df["hard"] = 1
    return df
#>


def prepare_pyg_datasets(df, output_dir, seed=1, n_procs=1, random_features_p=0.5, n_random_features=1):
    output_dir = join(output_dir, "pyg")
    os.makedirs(output_dir, exist_ok=True)
    print("using the same seed for all datasets:", seed)
    
    # no rf
    print(f"nodes with no RF - n={n_random_features}")
    pyg_ds = build_dimacs_representation(df, False, random_features_p=None, n_random_features=n_random_features, n_procs=n_procs, seed=seed)
    out_fn = join(output_dir, "pyg_dataset_no_rf.pt")
    torch.save(pyg_ds, out_fn)
    print("PyTorch Geometric dataset saved to", out_fn)
    print("PyTorch Geometric dataset length:", len(pyg_ds))

    # rf p=1.0
    print(f"all nodes with RF - n={n_random_features}")
    pyg_ds = build_dimacs_representation(df, True, random_features_p=1.0, n_random_features=n_random_features, n_procs=n_procs, seed=seed)
    out_fn = join(output_dir, "pyg_dataset_all_rf.pt")
    torch.save(pyg_ds, out_fn)
    print("PyTorch Geometric dataset saved to", out_fn)
    print("PyTorch Geometric dataset length:", len(pyg_ds))
    
    # rf p passed as an argument
    print(f"{random_features_p} nodes with RF - n={n_random_features}")
    pyg_ds = build_dimacs_representation(df, True, random_features_p=random_features_p, n_random_features=n_random_features, n_procs=n_procs, seed=seed)
    out_fn = join(output_dir, f"pyg_dataset_{random_features_p}_rf.pt")
    torch.save(pyg_ds, out_fn)
    print("PyTorch Geometric dataset saved to", out_fn)
    print("PyTorch Geometric dataset length:", len(pyg_ds))
#<


def main(args):
    # -- 0 --
    if args.rename == "yes":
        print("Renaming files in the input directory")
        rename_files(args.input_dir)  # , args.append_sat)

    # -- 1 -- 
    # collect all files in args.input_dir/[train|test|val]/[sat|unsat] and store them in a dict
    # with the key being the file name and the value being the file path
    file_dict = collect_files(args.input_dir)  
    
    # -- 2 --
    # save train, valid, test splits
    save_splits(file_dict, args.output_dir)
    
    # -- 3 --
    # produce a dataframe as the one from "full_data_generation" in "src/data/data_generation.py"
    df = prepare_dataframe(file_dict)
    out_fn = os.path.join(args.output_dir, "dataset.csv")
    df.to_csv(out_fn)
    print("Full dataset, length:", len(df))
    print(f"Dataset saved to {out_fn}")
    out_fn = out_fn.replace(".csv", ".pkl")
    df.to_pickle(out_fn)
    print(f"Dataset saved to {out_fn}") 

    # -- 4 --
    # to pyTorch Geometric
    df = pd.read_pickle(out_fn).reset_index()
    prepare_pyg_datasets(df, args.output_dir)
#<



# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G4SATBench data import")
    parser.add_argument("--input_dir", type=str, default="~/workspace/G4SATBench/dataset")
    parser.add_argument("--output_dir", type=str, default="../../input/dataset/g4satbench")
    parser.add_argument("--rename", type=str, default="no", choices=["yes", "no"], help="Rename files in the input directory")
    # parser.add_argument("--append_sat", type=str, default="no", choices=["yes", "no"], help="Append sat or unsat to the file name")
    args = parser.parse_args()
    
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