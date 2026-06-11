import argparse
import gc
import os
import neptune
import torch
import numpy as np

from utils.config import load_config
from data.data_generation import (
    full_data_generation,
)
from data.data_utils import prepare_dataset_splits # , load_dataset_splits, compute_stats, GnnCustomDataset2, compute_metrics
from learning import (
    do_train, 
    do_test
)

import random


def setup(conf):
    if not os.path.exists(conf["output_dir"]):
        os.makedirs(conf["output_dir"], exist_ok=True)
    
    if not os.path.exists(conf["dataset_dir"]):
        os.makedirs(conf["dataset_dir"], exist_ok=True)
    # if not os.path.exists(os.path.join(conf["output_dir"],conf["dataset_dir"])):
    #     os.makedirs(os.path.join(conf["output_dir"],conf["dataset_dir"]), exist_ok=True)
    # Set random seeds
    np.random.seed(int(conf["np_seed"]))
    torch.manual_seed(int(conf["torch_seed"]))
    random.seed(int(conf["rand_seed"]))
    torch.manual_seed(int(conf["torch_seed"]))
#<
    


def pipeline(conf):
    if conf["dev_mode"] or (not conf["remote_log"]):
        neptune_mode = "debug"
    else:
        neptune_mode = "async"
    
    print(f"Dev mode: {conf['dev_mode']}, remote log: {conf['remote_log']}")
    print(f"Neptune mode: {neptune_mode}")

    tags = [ "g4sat", conf.dataset ]
    tags.append( f"{conf['embedding_size']}" )


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
    args = parser.parse_args()
    config = load_config(args.config)
    setup(config)
    pipeline(config)