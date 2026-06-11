import argparse
import gc
import os
import neptune
import torch
import numpy as np

from utils.config import load_config
from data.data_generation import (
    full_data_generation,
    encode_data
)

from utils.misc import (
        clean_folder, 
 )

from data.data_utils import prepare_dataset_splits # , load_dataset_splits, compute_stats, GnnCustomDataset2, compute_metrics
from learning import (
    do_train, 
    do_test
)



import random


def setup(conf):
    # Set random seeds
    np.random.seed(int(conf["np_seed"]))
    torch.manual_seed(int(conf["torch_seed"]))
    random.seed(int(conf["rand_seed"]))
    torch.manual_seed(int(conf["torch_seed"]))
#<
    


def pipeline(conf):
    df = None
    ts_data_fld = conf["dataset_dir"]
    if conf["do_gen_data"]:
        if os.path.exists(ts_data_fld) and os.path.isdir(ts_data_fld):
            assert conf["force_overwrite"], f"output folder {ts_data_fld} already exists, use --force_overwrite to overwrite"
            clean_folder(ts_data_fld)  
        if not os.path.exists(ts_data_fld):
            os.makedirs(ts_data_fld, exist_ok=True)
        print("Generating dataset...")
        ts_data_fld = full_data_generation(conf)
        gc.collect()
    
    # update data_folder needed in case of timestamped output
    conf["dataset_dir"] = ts_data_fld

    if conf["do_encode_data"]:
        print("Enconding dataset (inluding random features, if requested)...")
        encode_data(conf)
        gc.collect()


    # if conf["do_gen_data"] or conf["do_split_data"]:
    if conf["do_split_data"]:
        if not os.path.exists(conf["output_dir"]):
            os.makedirs(conf["output_dir"], exist_ok=True)
        prepare_dataset_splits(conf)
        gc.collect()
    #< if do_gen_data or do_split_data

    # at this point we have the following files for training the GNN:
    # - {output_folder}/dataset_{exp_id}.pkl, selected examples from the full generated dataset {output_folder}/dataset.pkl
    # - {output_folder}/train_{exp_id}.txt.gz, list of formula_ids for the training set
    # - {output_folder}/val_{exp_id}.txt.gz, list of formula_ids for the validation set
    # - {output_folder}/test_{exp_id}.txt.gz, list of formula_ids for the test set

    if (not conf["do_train"]) and (not conf["do_test"]):
        print("No training or testing requested, exiting...")
        print("pipeline exits normally - all done")
        return


    if conf["dev_mode"] or (not conf["remote_log"]):
        neptune_mode = "debug"
    else:
        neptune_mode = "async"
    
    print(f"Dev mode: {conf['dev_mode']}, remote log: {conf['remote_log']}")
    print(f"Neptune mode: {neptune_mode}")

    tags = [ f"rf-{conf['add_random_features']}" ]
    if conf["add_random_features"]:
        tags.append( f"rf-p-{conf['random_features_p']}" )
    tags.append( f"var-{conf['min_vars']}-{conf['max_vars']}")
    # tags.append( f"cons-{conf['min_constr']}-{conf['max_constr']}")
    tags.append( f"emb-size:{conf['embedding_size']}" )

    exp_tags = conf.get("exp_tags", None)
    if exp_tags is not None:
        for tag in exp_tags.split(","):
            tags.append(tag.strip())
    
    run = neptune.init_run(
        project = conf["neptune_project"],
        mode = neptune_mode,
        # name = str(config["exp_id"]),
        description = str(config["exp_descr"]),
        tags = tags
    )

    run["args"] = conf
    run["configuration"].upload(args.config)

    if conf["do_train"]:
        print(" *** TRAINING ***")
        do_train(conf, run=run)
    
    if conf["do_test"]:
        print(" *** TESTING ***")
        do_test(conf, run=run)
    
  
    run.stop()
    print("pipeline exits normally - all done")
    #<


# ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--embedding_size', type=int, required=False, help='embedding size')
    parser.add_argument('--gcnpolicy_n_layers', type=int, required=False, help='number of gcn layers')
    args = parser.parse_args()
    config = load_config(args.config)
    # out_folder = config["output_dir"]
    # # if args contains emb_size and gcn_layers, update config
    
    # print("output folder:", out_folder)
    # print("embedding size:", config.embedding_size)
    # print("gnn layers:", config.gcnpolicy_n_layers)
    setup(config)
    pipeline(config)