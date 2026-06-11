
import os
from os.path import join
import gzip
import sys
import pandas as pd
from pulp import *
import os
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch


def prepare_dataset_splits(conf):    
    sel_df = select_examples(conf)
    stats_df = compute_stats(sel_df)
    print(f"method {conf['selection_method']}, selected subset of examples, len:", len(sel_df))
    print("stats on the selected examples")
    print(stats_df)

    splits = do_train_val_test_split(sel_df, conf)
    stats_dfs = []
    for k, l in splits.items():
        stat_df = compute_stats(sel_df[sel_df.formula_id.isin(l)])
        stat_df["split"] = k
        stat_df["size_split"] = len(l)
        # relative size of the split
        stat_df["p_split"] = len(l) / len(sel_df)
        stats_dfs.append(stat_df)
        print()
    stat_df = pd.concat(stats_dfs)
    stat_df = stat_df.reset_index().set_index(["split", "label", "hard"])
    print("stats on the splits")
    print(stat_df)
    
    # selection save in data_fld
    output_dir = conf["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    out_fn = join(output_dir, f"dataset_{conf['exp_id']}.pkl")
    sel_df.to_pickle(out_fn)
    print(f"dataset for exp {conf['exp_id']} saved to {out_fn}")
    out_fn = out_fn.replace(".pkl", ".csv")
    sel_df.to_csv(out_fn, index=False, sep=",")
    print(f"dataset for exp {conf['exp_id']} saved to {out_fn}")
    
    for split, ids in splits.items():
        out_fn = join(output_dir, f"{split}_{conf['exp_id']}.txt.gz")
        with gzip.open(out_fn, "wt") as f:
            f.write( "\n".join( map(str, ids) ) )
        print(f"{split} ids for exp {conf['exp_id']} saved to {out_fn}")
    #< for over splits
    out_fn = join(output_dir, f"split_stats_{conf['exp_id']}.csv")
    stat_df.to_csv(out_fn, sep=",")
    print(f"stats on the splits saved in: {out_fn}")
#< prepare_training_splits


def select_examples(conf):
    # select from the entire dataset a subset of examples given by "dataset_size" in the configuration
    data_folder = conf["dataset_dir"]
    pkl_fn = join(data_folder, "dataset.pkl")
    df = pd.read_pickle(pkl_fn)

    print(f"dataset read from {pkl_fn}")
    print("raw dataset shape:", df.shape)
    print(df.sample(1))

    method = conf["selection_method"]
    if method == "none":
        # raw dataset is returned
        return df
    elif method == "random":
        return select_samples_random(df, conf)
    elif method == "label_strat":
        return select_samples_label_strat(df, conf)
    elif method == "label_hard_strat":
        return select_samples_label_hard_strat(df, conf)
    else:
        assert False, f"unknown selection method: {method}"
#< select_examples


def select_samples_label_hard_strat(df, conf):
    # 4 x 2 numpy matrix with the values of (label, hard) and the number of examples available in df
    np_mat = np.zeros((2, 2), dtype=int)
    # 0, 0 -> label=0, hard=0
    np_mat[0, 0] = len(df[(df.label == 0) & (df.hard == 0)])
    # 0, 1 -> label=0, hard=1
    np_mat[0, 1] = len(df[(df.label == 0) & (df.hard == 1)])
    # 1, 0 -> label=1, hard=0
    np_mat[1, 0] = len(df[(df.label == 1) & (df.hard == 0)])
    # 1, 1 -> label=1, hard=1
    np_mat[1, 1] = len(df[(df.label == 1) & (df.hard == 1)])
    assert np.sum(np_mat) == len(df), "sum of np_mat != len(df)"
    
    # dataframe for pretty printing
    tmp_df = pd.DataFrame(np_mat, index=["neg", "pos"], columns=["easy", "hard"])
    print("dataset")
    print(tmp_df)
    del tmp_df

    # various checks for the number of examples
    #   the first check does not consider the sat/unsat among the hard examples
    dataset_size = conf["dataset_size"]
    n_req_hard = math.ceil(conf["hard_ratio"] * dataset_size)
    n_req_easy = conf["dataset_size"] - n_req_hard
    assert n_req_hard <= np.sum(np_mat[:, 1]), f"not enough hard examples, required / available (not considering sat): {n_req_hard} / {np.sum(np_mat[:, 1])}"
    assert n_req_easy <= np.sum(np_mat[:, 0]), f"not enough easy examples, required / available (not considering sat): {n_req_easy} / {np.sum(np_mat[:, 0])}"
    
    # fraction of positive (sat) vs negative (unsat) examples
    pos_p = conf["pos_ratio"]
    neg_p = 1.0 - pos_p
    print(f"From conf, required POS / NEG ratios: {pos_p} / {neg_p}")
    
    n_req_hard_p = math.ceil(n_req_hard * pos_p)
    n_req_hard_n = n_req_hard - n_req_hard_p
    n_req_easy_p = math.ceil(n_req_easy * pos_p)
    n_req_easy_n = n_req_easy - n_req_easy_p
    assert n_req_hard_p <= np_mat[1, 1], f"not enough hard pos examples, required / available: {n_req_hard_p} / {np_mat[1, 1]}"
    assert n_req_hard_n <= np_mat[0, 1], f"not enough hard neg examples, required / available: {n_req_hard_n} / {np_mat[0, 1]}"
    assert n_req_easy_p <= np_mat[1, 0], f"not enough easy pos examples, required / available: {n_req_easy_p} / {np_mat[1, 0]}"
    assert n_req_easy_n <= np_mat[0, 0], f"not enough easy neg examples, required / available: {n_req_easy_n} / {np_mat[0, 0]}"


    # randomly select examples from the easy and hard regions
    rng = np.random.default_rng(conf["data_seed"])
    easy_pos_df = df[(df.label == 1) & (df.hard == 0)].sample(n=n_req_easy_p, random_state=rng)
    easy_neg_df = df[(df.label == 0) & (df.hard == 0)].sample(n=n_req_easy_n, random_state=rng)
    hard_pos_df = df[(df.label == 1) & (df.hard == 1)].sample(n=n_req_hard_p, random_state=rng)
    hard_neg_df = df[(df.label == 0) & (df.hard == 1)].sample(n=n_req_hard_n, random_state=rng)
    # combine then shuffle using data_seed
    df = pd.concat([easy_pos_df, easy_neg_df, hard_pos_df, hard_neg_df]).sample(frac=1, random_state=rng).reset_index(drop=True)
    assert len(df) == dataset_size, f"final dataset size {len(df)} != {dataset_size}"
    return df
#< select_examples


# basic selection, no checks, no ratios, etc.
def select_samples_label_strat(df, meta_args):
    dataset_size = meta_args["dataset_size"]
    pos_label_p = meta_args["pos_ratio"]

    n_pos_req = math.ceil(dataset_size * pos_label_p)
    n_neg_req = dataset_size - n_pos_req
    n_pos_avail = len( df[df.label==1] )
    n_neg_avail = len( df[df.label==0] )
    
    print(f"n pos required / available: {n_pos_req} / {n_pos_avail}")
    print(f"n neg required / available: {n_neg_req} / {n_neg_avail}")

    assert (n_pos_req <= n_pos_avail) and (n_neg_req <= n_neg_avail), f"not enough data for requested split"
  
    rng = np.random.default_rng(meta_args["data_seed"])
    # select n_pos_req from label==1 examples
    pos_df = df[df.label==1].sample(n=n_pos_req, random_state=rng)
    neg_df = df[df.label==0].sample(n=n_neg_req, random_state=rng)
    # combine then shuffle using data_seed
    df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=rng).reset_index(drop=True)
    print("final dataset shape:", df.shape)   
    assert len(df) == dataset_size, f"final dataset size {len(df)} != {dataset_size}"
    assert len(df[df.label==1]) == n_pos_req, f"final pos size {len(df[df.label==1])} != {n_pos_req}"
    assert len(df[df.label==0]) == n_neg_req, f"final neg size {len(df[df.label==0])} != {n_neg_req}"
    return df
#< select_samples_basic


# basic selection, no checks, no ratios, etc.
def select_samples_random(df, meta_args):
    dataset_size = meta_args["dataset_size"]
    return df.sample(n=dataset_size, random_state=meta_args["data_seed"])
#< select_samples_basic


def compute_stats(df, group_cols=["label", "hard"]):
    res = df.groupby(group_cols).agg(
        split_size=("formula_id", "count"),
        p=("formula_id", lambda x: len(x) / len(df)),
        n_var_min=("n_var", "min"),
        n_var_max=("n_var", "max"),
        n_clauses_min=("n_clauses", "min"),
        n_clauses_max=("n_clauses", "max"),
    )
    return res
    
    
    # counts_df = df.groupby(group_cols).size()
    # percs_df = df.groupby(group_cols).size() / len(df)
    # # join the two pd.Series in a dataframe
    # stats_df = pd.concat([counts_df, percs_df], axis=1)
    
    # stats_df.columns = ["count", "p"]
    # return stats_df
#< compute_stats

    
def compute_stats_old(df, group_cols=["label", "hard"]):
    counts_df = df.groupby(group_cols).size()
    percs_df = df.groupby(group_cols).size() / len(df)
    # join the two pd.Series in a dataframe
    stats_df = pd.concat([counts_df, percs_df], axis=1)
    
    stats_df.columns = ["count", "p"]
    return stats_df
#< compute_stats



def do_train_val_test_split(df, conf):
    method = conf["splitting_method"]
    if method == "basic":
        return do_train_val_test_split_basic(df, conf)
    elif method == "stratify":
        return do_train_val_test_split_stratify(df, conf)
    else:
        assert False, f"unknown split method: {method}"


def do_train_val_test_split_stratify(df, conf):
    test_size = conf["test_size"]
    val_size = conf["val_size"]

    # split with a stratification on ["label", "hard"] returning the list of formula_ids in the split
    te_size = test_size if isinstance(test_size, int) else math.ceil(len(df) * test_size)
    va_size = val_size if isinstance(val_size, int) else math.ceil(len(df) * val_size)
    tr_size = len(df) - te_size - va_size
    
    print("REQUESTED SIZES OF THE SPLITS")
    print(f"train size: {tr_size}, val size: {va_size}, test size: {te_size}")

    assert te_size + va_size + tr_size == len(df), "split sizes do not match, check your values"

    seed = conf["split_seed"]

    # set grouping columns: column hard is excluded when hard_ratio=-1 (ignored) or when hard_ratio=1 or hard_ratio=0
    if "stratify_cols" in conf:
        group_cols = [c.strip() for c in conf["stratify_cols"].split(",")]
    else:
        group_cols = ["label", "hard"] if ( (conf["hard_ratio"]!=1.0) or (df.hard.nunique() > 1)) else ["label"]
    
    print()
    print(f"stratifying on {len(group_cols)} columns: {group_cols}")

    # train,val / test split
    train_ids, test_ids = train_test_split(df["formula_id"], test_size=te_size, stratify=df[group_cols], random_state=seed)
    # train / val split
    if va_size > 0:
        # See https://stackoverflow.com/questions/79595277/stratification-fails-in-train-test-split
        train_df = df[df.formula_id.isin(train_ids)]
        train_ids, valid_ids = train_test_split(train_df.formula_id, test_size=va_size, stratify=train_df[group_cols], random_state=(seed + 1) % sys.maxsize)
    else:
        valid_ids = []

    # stats on train_ids
    # print()
    # print(f"TRAINING SET, size: {len(train_ids)}")
    # print(compute_stats(df.loc[df["formula_id"].isin(train_ids)], group_cols=group_cols))
    # print()
    # print(f"VALIDATION SET, size: {len(valid_ids)}")
    # print(compute_stats(df.loc[df["formula_id"].isin(valid_ids)], group_cols=group_cols))
    # print()
    # print(f"TEST SET, size: {len(test_ids)}")
    # print(compute_stats(df.loc[df["formula_id"].isin(test_ids)], group_cols=group_cols))
    
    return {"train": train_ids, "val":valid_ids, "test":test_ids}
#<


def do_train_val_test_split_basic(df, conf):
    # no checks here, do not use BASIC splitting if you do not want any failures in splitting
    
    test_size = conf["test_size"]
    val_size = conf["val_size"]

    test_size = test_size if isinstance(test_size, int) else math.ceil(conf["test_size"] * len(df))
    val_size = val_size if isinstance(val_size, int) else math.ceil(conf["val_size"] * len(df))
    train_size = len(df) - test_size - val_size

    # shuffle the dataframe randomly
    df = df.sample(frac=1, random_state=conf["split_seed"]).reset_index(drop=True)
    train_ids = df.iloc[:train_size]["formula_id"].tolist()
    if val_size > 0:
        val_ids = df.iloc[train_size:(train_size + val_size)]["formula_id"].tolist()
    else:
        val_ids = []

    test_ids = df.iloc[-test_size:]["formula_id"].tolist()
    return {"train": train_ids, "val":val_ids, "test":test_ids}
#<


import gc
def load_dataset_splits(meta_args, splits=["train", "val", "test"]):
    out_fld = meta_args["output_dir"]
    data_fld = meta_args["dataset_dir"]
    exp_id = meta_args["exp_id"]
    dataset_fn = join(data_fld, f"dataset_{exp_id}.pkl")
    pyg_data_fld = join(data_fld, "pyg")

    # df = pd.read_pickle(dataset_fn)
    dataset_fn = build_enc_dataset_filename(meta_args)
    dataset_fn = join(pyg_data_fld, dataset_fn)
    torch_ds = torch.load(dataset_fn, weights_only=False)
    print("full torch (pyg) dataset read from:", join(pyg_data_fld, dataset_fn))
    print("\t examples:", len(torch_ds))
    for i, (k, v) in enumerate(torch_ds[0].items()):
        print(f"{i+1}) \t{k}: {type(v)}", end="")
        if torch.is_tensor(v):
            print(f", {v.size()}, {v.dtype}, value={v}")
        else:
            print()
   
    torch_ds2 = {int(torch_ds[i].formula_id): torch_ds[i] for i in range(len(torch_ds))}
    print("torch dataset size:", len(torch_ds2))
    out = defaultdict(list)
    for split in splits:
        split_fn = join(out_fld, f"{split}_{exp_id}.txt.gz")
        with gzip.open(split_fn, "rt") as f:
            ids = map(int, f.read().splitlines())
        
        m = [torch_ds2[e] for e in ids]
        print(f"{split} size:", len(m))

        # select rows with formula_id in ids from the hugging face dataset gr_df
        # filt_gr_df = gr_df.filter(lambda row: row["formula_id"] in ids).to_pandas()
        # filt_gr_df = gr_df.filter(lambda row: np.isin(row['formula_id'], ids), batched=False)
        
        # m = m.merge(gr_df, on="formula_id", how="left")
        out[split] = m
    torch_ds2 = None
    # force collector
    gc.collect()
    # print info on available RAM memory
    
    return out
#<

import gc
def load_dataset_splits_g4s(meta_args, splits=["train", "val", "test"]):
    out_fld = meta_args["output_dir"]
    data_fld = meta_args["dataset_dir"]
    exp_id = meta_args["exp_id"]
    # dataset_fn = join(data_fld, f"dataset_{exp_id}.pkl")
    pyg_data_fld = join(data_fld, "pyg")

    # df = pd.read_pickle(dataset_fn)
    dataset_fn = meta_args["dataset"]
    torch_ds = torch.load(join(pyg_data_fld, dataset_fn), weights_only=False)
    
    print("full torch (pyg) dataset read from:", join(pyg_data_fld, dataset_fn))
    
    print("\t examples:", len(torch_ds))
    for i, (k, v) in enumerate(torch_ds[0].items()):
        print(f"{i+1}) \t{k}: {type(v)}", end="")
        if torch.is_tensor(v):
            print(f", {v.size()}, {v.dtype}, value={v}")
        else:
            print()
   
    torch_ds2 = {int(torch_ds[i].formula_id): torch_ds[i] for i in range(len(torch_ds))}
    print("torch dataset size:", len(torch_ds2))
    out = defaultdict(list)
    for split in splits:
        split_fn = join(data_fld, f"{split}.txt.gz")
        with gzip.open(split_fn, "rt") as f:
            ids = map(int, f.read().splitlines())
        
        m = [torch_ds2[e] for e in ids]
        labels = defaultdict(int)
        for e in m:
            labels[int(e.label)] += 1
        for k, v in labels.items():
            print(f"{split} label {k}: {v}")
        print(f"{split} size:", len(m))

        # select rows with formula_id in ids from the hugging face dataset gr_df
        # filt_gr_df = gr_df.filter(lambda row: row["formula_id"] in ids).to_pandas()
        # filt_gr_df = gr_df.filter(lambda row: np.isin(row['formula_id'], ids), batched=False)
        
        # m = m.merge(gr_df, on="formula_id", how="left")
        out[split] = m
    torch_ds2 = None
    # force collector
    gc.collect()
    # print info on available RAM memory
    
    return out



from torch_geometric.data import Dataset, Data

class GraphDataset(Dataset):
    def __init__(self, examples):
        super(GraphDataset).__init__()
        self.examples = examples
            
    def len(self):
        return len(self.examples)
    
    def get(self, idx):
        return self.examples[idx]
#< GraphDataset


def build_enc_dataset_filename(conf):
    n_random_features, random_features_p, random_features_seed = (0, 0, 0) if not conf["add_random_features"] \
        else (conf["n_random_features"], conf["random_features_p"], conf["random_features_seed"])

    filename = f"rf_{conf['add_random_features']}_n{n_random_features}_p{random_features_p}_s{random_features_seed}_pyg_dataset.pt"
    return filename