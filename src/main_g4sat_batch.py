import argparse
import os
import numpy as np
import subprocess
from os.path import join
from itertools import product
import yaml



# GRID
embedding_size = [32, 64, 128]

gcn_layers = [2, 16, 32]

pyg_filenames = ["pyg_dataset_no_rf.pt", "pyg_dataset_0.5_rf.pt",  "pyg_dataset_all_rf.pt"]


# ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True, help='Path to folder with config files')
    args = parser.parse_args()
    
    if os.path.exists(args.config_dir) and os.path.isdir(args.config_dir):
        conf_files = sorted([join(args.config_dir, f) for f in os.listdir(args.config_dir) if f.endswith('.yaml')])
    else:
        conf_files = [args.config_dir]

    print(f"Found {len(conf_files)} config files in {args.config_dir}:")
    
    for c in conf_files:
        print(c)
    print("--")
    print("confirm?")
    user_input = input("y to accept\n")
    if user_input.lower() != 'y':
        print("Exiting...")
        exit(0)

    params = [{"dataset": ds_fn, "embedding_size": es, "gcn_layers": gl} for ds_fn, es, gl in product(pyg_filenames, embedding_size, gcn_layers)]

    for fn in conf_files:
        print(f"Running config: {fn}")
        # read yaml
        with open(fn, 'r') as f:
            orig_conf = yaml.safe_load(f)
        # run the command
        for p in params:
            conf = orig_conf.copy()
            embedding_size = p["embedding_size"]
            gcn_layers = p["gcn_layers"]
            
            conf["dataset"] = p["dataset"]
            conf["embedding_size"] = embedding_size
            conf["gcnpolicy_n_layers"] = gcn_layers

            # take last part of output_dir
            out_folder = conf["output_dir"]
            out_folder = os.path.basename(out_folder)
            out_folder = p["dataset"].rsplit(".", 1)[0] + "-" + out_folder + f"-e{embedding_size}-l{gcn_layers}"
            out_folder = os.path.join(conf["output_dir"], out_folder)
            conf["output_dir"] = out_folder
            # check if the file test-results.csv exists in output folder
            if os.path.exists(join(out_folder, "test-results.csv")):
                print(f"Skipping {out_folder} as test-results.csv already exists")
                continue
            
            os.makedirs(out_folder, exist_ok=True)
            # save config file
            out_conf = join(out_folder, os.path.basename(fn))
            with open(out_conf, 'w') as f:
                yaml.dump(conf, f)

            print("Dataset: ", p["dataset"])
            print(f"Running with embedding_size={embedding_size} and gcn_layers={gcn_layers}")
            print("output folder: ", out_folder)
            print("config: ", out_conf)
            # run the command
            subprocess.run(["python", "main_g4sat.py", "--config", out_conf])
            print("subprocess terminated, output in ", out_folder)
        #< for over param grid
    #< for over config files
    print("all done.")
#<
