import os
from os.path import join
import sys
from tqdm import tqdm
import pandas as pd
from pulp import *
from typing import List, Tuple, Dict
from datetime import datetime
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import humanize as H
import math
from collections import defaultdict
import torch
import time
from joblib import Parallel, delayed

from utils.misc import (
        clean_folder, 
        get_n_procs )



import cnfgen
from torch_geometric.data import (
    Data, 
    HeteroData
)

from data.solving import solve_SAT_problem2
from data.mps import convert_dimacs
from data.data_utils import build_enc_dataset_filename

# from milp.utils.utils import clean_folder, get_n_procs

# from milp.utils.milp2vec import milp_to_vec
# from milp.utils.mps import convert_dimacs


def full_data_generation(conf):
    gen_version = int(conf["generation_version"])
    assert gen_version in [1, 2, 3], "generation_version must 1,2,3"

    generate_size = conf["generate_size"]
    n_procs = get_n_procs(conf["data_gen_n_procs"])
    
    print(f"Data generation (version {gen_version}). Total size (before selection): {generate_size}, using n_procs={n_procs}")

    output_dir=conf['dataset_dir']
    timestamped_subdir = False
    if timestamped_subdir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
    else:
        output_dir = output_dir

    #<

    os.makedirs(output_dir, exist_ok=True)
    print(f"saving data to {output_dir}")

    time1 = time.perf_counter()
    if gen_version == 3:
        df = generate_dataset_v3(
            size = generate_size,
            data_seed=int(conf["data_seed"]),
            min_n_var=int(conf["min_vars"]),
            max_n_var=int(conf["max_vars"]),
            nc_left=conf["nc_left"],
            nc_right=conf["nc_right"],
            k=int(conf['k_cnf']),
            output_dir=output_dir,
            n_procs=n_procs,
            fail_with_duplicates=conf["fail_with_duplicates"]
        )
    elif gen_version == 2:
        df = generate_dataset_v2(
            size = generate_size,
            data_seed=int(conf['data_seed']),
            n_var=int(conf['max_vars']),
            k=int(conf['k_cnf']),
            max_clauses=int(conf['max_clauses']),
            output_dir=output_dir,
            n_procs=n_procs,
            method="fast"
        )
    else:
        df = generate_dataset_v1()
    
    time2 = time.perf_counter()
    print(f"data generation completed in {H.precisedelta(time2 - time1)}")
    



    print(f"full dataset (before selection), shape: {df.shape}")
    print(f"min_n_var: {conf['min_vars']}")
    print(f"max_n_var: {conf['max_vars']}")
    print(f"unique number of vars in generated dataset: {len(df.n_var.unique())}")
    print(f"min / max n_clauses in generated dataset: {df.n_clauses.min()} / {df.n_clauses.max()}")

    time3 = time.perf_counter()
    dataset = Dataset.from_pandas(df)
    if "data_solve_n_procs" in conf:
        n_procs_solve = get_n_procs(conf["data_solve_n_procs"])
    else:
        n_procs_solve = n_procs
    
    dataset = dataset.map(lambda row: solve_SAT_problem2(row), batched=True, num_proc=n_procs_solve, desc="solving")
    time4 = time.perf_counter()
    print("solving took", H.precisedelta(time4 - time3))

    df = dataset.to_pandas()
    df = add_is_hard_col(df)
    
    print("data grouped by label:")
    print(df.groupby("label").size())
    print()
    print("data grouped by hard:")
    print(df.groupby("hard").size())
    print()
    print("data grouped by label and hard:")
    print(df.groupby(["label", "hard"]).size())
    print()
    print("distribution")
    print(df[["label", "hard"]].value_counts(normalize=True))

    # for each n_var show the list of n_clauses
    df_by_n_var = df.groupby("n_var").agg(
        unique_n_clauses=("n_clauses", "nunique"),
        min_n_clauses=("n_clauses", "min"),
        max_n_clauses=("n_clauses", "max"),
        unique_n_labels=("label", "nunique")
    )
    print(df_by_n_var)

    df_by_label = df.groupby("label").agg(
        unique_n_clauses=("n_clauses", "nunique"),
        min_n_clauses=("n_clauses", "min"),
        max_n_clauses=("n_clauses", "max"),
        unique_n_var=("n_var", "nunique"),
        min_n_vars=("n_var", "min"),
        max_n_vars=("n_var", "max")
    )
    print(df_by_label)

    print("saving dataset")
    time1 = time.perf_counter()
    out_fn = os.path.join(output_dir, "dataset.csv")
    df.to_csv(out_fn, index=False, sep=",")
    print(f"dataset saved to {out_fn}")
    out_fn = out_fn.replace(".csv", ".pkl")
    df.to_pickle(out_fn)
    print(f"dataset saved to {out_fn}")
    return output_dir
#<



def encode_data(conf):
    # read dataset
    filename = os.path.join(conf["dataset_dir"], "dataset.pkl")
    # df = pd.read_pickle(filename)
    # TODO:
    filename = filename.replace(".pkl", ".csv")
    df = pd.read_csv(filename, sep=",")
    print(f"dataset loaded from {filename}")
    print(f"dataset shape: {df.shape}")
    
    # 
    time5 = time.perf_counter()
    print("generating graph representations (parallel)")
    print("tensors extended with random features?", conf["add_random_features"] )
    
    n_rdm_feats = conf['n_random_features'] if conf["add_random_features"] else None
    seed = conf.get("random_features_seed", None)
    add_rf_features = conf["add_random_features"]
    random_features_p = conf.get("random_features_p", None)    
    n_procs = conf.get("data_gen_n_procs", 1)
    
    gr_df = build_dimacs_representation(df, add_random_features=add_rf_features, random_features_p=random_features_p, 
                                        n_random_features=n_rdm_feats, n_procs=n_procs, seed=seed)
    time6 = time.perf_counter()
    print(f"graph repr computed in {H.precisedelta(time5 - time6)} (parallel)")
    # print(f"graph repr, len={len(gr_df)}, type={type(gr_df)}, type(1st el)= {type(gr_df[0])}, type(x)= {type(gr_df[0]['x'])}")

    time1 = time.perf_counter()
    os.makedirs(join(conf["dataset_dir"], "pyg"), exist_ok=True)
    filename = build_enc_dataset_filename(conf)
    out_fn = join(conf["dataset_dir"], "pyg", filename)  # "pyg_dataset.pt")
    torch.save(gr_df, out_fn)
    print(f"graph reprentations (pyg) saved to {out_fn}")
    time2 = time.perf_counter()
    print("saving took", H.precisedelta(time2 - time1))
#<


def generate_dataset_v3(size,
                     data_seed,
                     min_n_var, max_n_var,
                     nc_left, nc_right,
                     k,
                     output_dir,
                     n_procs=1,
                     fail_with_duplicates=True):
    assert min_n_var is not None, "Min number of variables must be specified."
    assert max_n_var is not None, "Min number of variables must be specified."
    
    data_seed = data_seed if data_seed > -1 else int(datetime.now().timestamp())
    print(f"using data seed: {data_seed}")

    rng = np.random.default_rng(seed = data_seed)
    seeds = rng.integers(0, sys.maxsize, size)
    
    formula_id = [i+1 for i in np.arange(size)]
    n_vars = rng.integers(min_n_var, max_n_var+1, size)
    print(f"random number of vars (one per formula) generated, min={np.min(n_vars)}, max={np.max(n_vars)}, median={np.median(n_vars)}")
    
    print("Using Crawford & Auton, 1996 for n_clauses")
    n_clauses_hard = map(compute_crawford_auton, n_vars)
    
    if (nc_left > 0) or (nc_right > 0):
        print("Adjusting number of clauses: random extraction from interval (no fixed hard n clauses)")
        def from_interval(left, right):
            s = rng.integers(left, right+1)
            # print(f"from_interval: {left} - {right} -> {s}")
            return s
        # print("nc_left", nc_left)
        # print("nc_right", nc_right)
        # print("Modifying number of clauses hard")

        def adjust_left(x):
            adj = x - nc_left if isinstance(nc_left, int) else math.floor(x - x * nc_left)
            # print(f"LEFT: {x} -> {adj}  {nc_left}")
            return adj

        def adjust_right(x):
            adj = x + nc_right if isinstance(nc_right, int) else math.floor(x + x * nc_right)
            # print(f"RIGHT: {x} -> {adj}  {nc_right}")
            return adj

        def compute_adj_range(x):
            left = adjust_left(x)
            right = adjust_right(x)
            return (left, right)

        # adj_ranges = map(compute_adj_range, n_clauses_hard)
        adj_ranges = [compute_adj_range(n) for n in tqdm(n_clauses_hard)]
        # n_clauses_hard = map(lambda x: from_interval(x[0], x[1]), adj_ranges)
        n_clauses_hard = [from_interval(x[0], x[1]) for x in tqdm(adj_ranges)]
    else:
        assert False, "check left right"
    
    n_clauses = [int(round(n)) for n in n_clauses_hard]

    df = pd.DataFrame({'formula_id':formula_id, 'seed': seeds, 'k': k, 'n_var': n_vars, 'n_clauses': n_clauses})
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(batch_gen, batched=True, num_proc=n_procs, fn_kwargs={"output_dir":output_dir}, desc="generating formulae (v3)")

    # compute hash for formula_repr
    df = dataset.to_pandas()
    
    n_unique_hashes = len(df.formula_hash.unique())
    print(f"number of unique formulae: {n_unique_hashes} out of {len(df)}")
    if n_unique_hashes < len(df):
        print("WARNING: there are duplicates in the generated dataset")
        if fail_with_duplicates:
            assert False, "duplicates in the generated dataset"
    

    df["folder"] = output_dir
    return df
#< generate_data_v3


def generate_dataset_v2(size,
                     data_seed,
                     n_var,
                     k,
                     var_multiplier_pm,
                     output_dir,
                    var_multiplier,
                     max_clauses = None,
                     n_procs=1
                     ):
    assert n_var is not None, "Number of variables must be specified."

    data_seed = data_seed if data_seed > -1 else int(datetime.now().timestamp())
    print(f"using data seed: {data_seed}")
    rng = np.random.default_rng(seed = data_seed)
    seeds = rng.integers(0, sys.maxsize, size)

    formula_id = [i+1 for i in np.arange(size)]
    max_clauses = max_clauses if max_clauses is not None else round(compute_crawford_auton(n_var))

    # below, _max_clauses inclusive
    n_clauses = rng.integers(1, max_clauses+1, size)
    if var_multiplier > 0:
        n_clauses = map(lambda x: rng.integers(math.floor(x - var_multiplier_pm * x), math.ceil(x + var_multiplier_pm * x)), n_clauses)
    
    # dataframe with columns 'seed' and 'n_clauses'
    df = pd.DataFrame({'formula_id':formula_id, 'seed': seeds, 'k': k, 'n_var': n_var, 'n_clauses': n_clauses})
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(batch_gen, batched=True, fn_kwargs={"output_dir":output_dir}, num_proc=n_procs, desc="generating formulae (v2)")
    
    df = dataset.to_pandas()
    df["folder"] = output_dir
    return df
#<


def generate_dataset_v1(size,
                     data_seed,
                     n_var,
                     k,
                     max_clauses,
                     output_dir,
                     n_procs=1): 
    data_seed = data_seed if data_seed > -1 else int(datetime.now().timestamp())

    rng = np.random.default_rng(seed = data_seed)
    seeds = rng.integers(0, sys.maxsize, size)
    
    formula_id = [i+1 for i in np.arange(size)]
    
    df = pd.DataFrame({'formula_id':formula_id, 'seed': seeds, 'k': k, 'n_var': n_var, 'n_clauses': max_clauses})
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(batch_gen, batched=True, fn_kwargs={"output_dir":output_dir}, num_proc=n_procs, desc="generating formulae (v1)")
    
    df = dataset.to_pandas()
    df["folder"] = output_dir
    
    return df
#<


def generate_formula(kcnf, number_of_vars, number_of_clauses, seed):
    return cnfgen.RandomKCNF(kcnf, number_of_vars, number_of_clauses, seed=seed)
#<


def get_dimacs_format(formula: cnfgen.CNF) -> str:
    """Convert formula to DIMACS format."""
    # Start with DIMACS format string
    lines = []
    
    # Get clauses from the formula
    for cls in formula:
        # Convert each clause to list of integers
        clause = [str(int(lit)) for lit in cls]
        lines.append(" ".join(clause) + " 0")
    #<
    # sort lines
    lines.sort() # this is in-place
    lines = [f"p cnf {formula.number_of_variables()} {len(formula)}"] + lines 
    return "\n".join(lines)
#<


def save_formula(formula, filepath):
        representation = get_dimacs_format(formula)  # this is a string
        with open(filepath, 'w') as f:
            f.write(representation)
        # returh hash of representation

        return hash(representation)
#<

#> HF batch processing
def batch_gen(batch, output_dir):
    output = defaultdict(list)
    for fi, seed, nc, nv, nk in zip(batch['formula_id'], batch['seed'], batch['n_clauses'], batch['n_var'], batch['k']):
        fil_name = f"{fi}_vars{nv}_k{nk}_clauses{nc}.cnf"
        output["formula_hash"].append(save_formula(generate_formula(nk, nv, nc, seed), os.path.join(output_dir, fil_name)))
        output['filename'].append(fil_name)
    #< for
    return output
#< def batch_gen

minus_two_thirds = - 2.0 / 3.0
def compute_crawford_auton(n_vars):
        # Crawford & Auton, "Experimental results on the crossover point in random 3-SAT", 1996
        # m = 4.258 n + 58.26 nˆ{−2/3} 
        
        a = 4.258 * n_vars
        b = 58.26 * ( n_vars ** minus_two_thirds )
        res = int(a + b)

        # idxs = [0, 1, 2]
        # for i in idxs:
        #     print(f"{n_vars[i]}, a={a[i]}, b={b[i]}, n_clauses={res[i]}")

        return res
#<

def add_is_hard_col(df, w_ext_p=0):
    df_hard = df.groupby("n_clauses")["label"].nunique().reset_index()
    df_hard.columns = ["n_clauses", "n_labels"]
    # h a r d    r e g i o n
    m_h = df_hard[df_hard["n_labels"] > 1].n_clauses.min()
    M_h = df_hard[df_hard["n_labels"] > 1].n_clauses.max()
    w_h = M_h - m_h
    print(f"hard region: [{m_h}, {M_h}], width: {w_h}")

    ext_w_h = math.ceil(w_h * w_ext_p)
    ext_w_h_2 = math.ceil(ext_w_h/2)
    m_h_ext = m_h - ext_w_h_2
    M_h_ext = M_h + ext_w_h_2
    print(f"\textended hard region (if requested): [{m_h_ext}, {M_h_ext}], ext width {w_h + ext_w_h}")
    print(f"\t  fraction: {w_ext_p}, width: {ext_w_h}")

    df["hard"] = df["n_clauses"].apply(lambda x: 1 if m_h_ext <= x <= M_h_ext else 0)
    return df
#<


def build_dimacs_representation(df, add_random_features, random_features_p=None, n_random_features=None, n_procs=1, seed=None):
    if add_random_features:
        assert random_features_p is not None, "random_features_p must be specified if add_random_features is True"
        assert n_random_features is not None, "n_random_features must be specified if add_random_features is True"
        assert seed is not None, "seed must be specified if add_random_features is True"
        ss = np.random.SeedSequence(seed)
        gen_ss = ss.spawn(len(df))
        # use same seed in the following list to add the same random features to all the formulae
        rngs = [np.random.default_rng(s) for s in gen_ss]
    else:
        rngs = [None] * len(df)
    
    # rngs always defined here

    def build_pyg_data(row, rng=None):
        formula_id, folder, filename, label, hard = row["formula_id"], row["folder"], row["filename"], row["label"], row["hard"]
        problem = convert_dimacs(join(folder, filename), save_to_file=False)
        if add_random_features:
            hdata = milp_to_graph(problem, add_random_features, random_features_p, n_random_features, rng=rng) #FIXME: the last graph index will be input, to be used to create a graph
        else:
            hdata = milp_to_graph(problem, add_random_features=False)

        hdata.formula_id = torch.tensor(formula_id, dtype=torch.long)
        hdata.label = torch.tensor(label, dtype=torch.float)  # float(label)
        hdata.hard = torch.tensor(hard, dtype=torch.long)
        return hdata
    #< build_py_data

    # create random seeds for the spawn threads
    pbar = tqdm(df.iterrows(), total=len(df), desc=f"to pyg data (n_procs {n_procs})", leave=True)
    results = Parallel(n_jobs=n_procs)(delayed(build_pyg_data)(row, rng) for (_, row), rng in zip(pbar, rngs)) 
    pbar.close()

    e = results[0]
    print(f"torch dataset completed, len={len(results)}, features and types:")
    for i, (k, v) in enumerate(e.items()):
        print(f"{i+1}) \t{k}: {type(v)}", end="")
        if torch.is_tensor(v):
            print(f", {v.size()}")
        else:
            print()

    return results
#<

# GEOMETRIC DATA
#  -- bipartite: homogeneous nodes, cannot compute "group" pooling ops
#  -- hetero: heterogeneous nodes, can compute "group" pooling ops (current code based on this)
additional_tensor_keys = ["formula_id", "label", "hard"]
class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            inc = torch.tensor([[self.x_left.size(0)], [self.x_right.size(0)]])
        else:
            inc = super().__inc__(key, value, *args, **kwargs)
        # print(f"__inc__: {key} -> {inc}")
        return inc
    #<

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in additional_tensor_keys:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    @property
    def num_nodes(self):
        return self.x_left.size(0) + self.x_right.size(0)
    #<
#<

class BipartiteHeteroData(HeteroData):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in additional_tensor_keys:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
#<



# class LogicData(Data):
#     NON_TENSOR_FIELDS = ["label", "hard", "formula_id"]
#     def __init__(self, **kwargs):
#         super(LogicData, self).__init__(**kwargs)
        
#     def __cat_dim__(self, key, value, *args, **kwargs):
#         if key in LogicData.NON_TENSOR_FIELDS:
#             print("non tensor")
#             return None
#         return super().__cat_dim__(key, value, *args, **kwargs)


# NOTE: after doign first experiment without rand feature, add the rand features and double check the results an compare accuracy
# TODO: c--I ARGUMENTS???
def vec_to_graph(
    A: np.ndarray,                     # Constraint matrix (m x n)
    b: np.ndarray,                     # RHS vector (m)
    operators: List[str],       # List of constraint types ('<=', '=', '>=')
    add_random_features,
    random_features_p,
    n_random_features=1,
    rng=None,
    c: np.ndarray = None,             # Optional objective coefficients (n)
    l: np.ndarray = None,             # Optional lower bounds (n)
    u: np.ndarray = None,             # Optional upper bounds (n)
    I: List[int] = None,               # Optional indices of integer variables
) -> Data:
    """
    Convert a MILP to graph representation following Section 2.1 of the paper.
    """
    if add_random_features:
        assert rng is not None, "You must provide value for rng (random) if add_random_features is True"
        assert n_random_features is not None, "n_random_features must be set if add_random_features is True"
    
    m, n = A.shape  # m constraints, n variables
    # Initialize default values if not provided
    if c is None:
        c = np.zeros(n)
    if l is None:
        l = np.zeros(n)  # For binary problems, lower bound is 0
    if u is None:
        u = np.ones(n)   # For binary problems, upper bound is 1
    if I is None:
        I = list(range(n))  # For binary problems, all variables are integer
        
    # TODO: ??? Random features?
    # generate two randome vectors of size n and m
    
    # ORIGINAL CODE from: https://github.com/liujl11git/GNN-MILP/blob/main/1_generate_data.py

    # if is_append_rand:
	# 	print('Before appending:',ConFeatures_feas.shape,VarFeatures_feas.shape)
	# 	kkk = ConFeatures_feas.shape[0] // m
	# 	np.random.seed(0)
	# 	ConAug = np.tile(np.random.rand(m,1), (kkk,1))
	# 	VarAug = np.tile(np.random.rand(n,1), (kkk,1))
	# 	ConFeatures_feas = np.concatenate((ConFeatures_feas, ConAug),axis=1)
	# 	VarFeatures_feas = np.concatenate((VarFeatures_feas, VarAug),axis=1)
	# 	print('After appending:',ConFeatures_feas.shape,VarFeatures_feas.sha

    
    if add_random_features:
        r = rng.random()
        if rng.random() < random_features_p:
            # generated m x n_random_featurs random elements using rng
            rand_vec_m = rng.random((m, n_random_features))
            rand_vec_n = rng.random((n, n_random_features))
            # print("shape of rand_vec_m", rand_vec_m.shape)
            # print("shape of rand_vec_n", rand_vec_n.shape)
        else:
            rand_vec_m = np.zeros((m, n_random_features))
            rand_vec_n = np.zeros((n, n_random_features))
        
    
    # CONSTRAINTS  V
    # (bi, op)
    # OLD COMMENT: constraint_type_map = {'<=': -1, '=': 0, '>=': 1}
    # OLD COMMENT: constraint_type_values = [constraint_type_map[ct] for ct in operators]
    V_features = np.zeros((m, 2))
    V_features[:, 0] = b
    V_features[:, 1] = operators
    if add_random_features:
        #print("V_features.shape:", V_features.shape)
        #print("rand_vec_m.shape:", rand_vec_m.shape)
        V_features = np.hstack( (V_features, rand_vec_m) )
        # print("V_features.shape:", V_features.shape)
    # VARIABLES  W
    # 1. Create node features
    # For variable nodes: (cj, lj, uj, τj)
    tau = np.zeros(n)
    tau[I] = 1
    W_features = np.column_stack([c, l, u, tau])
    if add_random_features:
        W_features = np.hstack( (W_features, rand_vec_n) )
        
    
    # W_features = np.column_stack([c, l, u, tau, rand_vec_n]) # add rand feature
    # Combine all node features

    # EDGES
    edge_index = [[], []]
    edge_attr = []    
    for i in range(m):  # for i over constraints
        for j in range(n):  # for j over variables
            if A[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_attr.append(A[i, j])
    
    # Convert to PyTorch tensors
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(edge_attr)
    const_f = torch.FloatTensor(V_features)
    var_f = torch.FloatTensor(W_features)
    

    # x_left, x_right, edge_index, **kwargs
    # res = BipartiteData(
    #     x_left = const_f,
    #     x_right = var_f,
    #     #
    #     edge_index=edge_index,
    #     x_edge=edge_attr,
    #     #
    #     # num_nodes=torch.tensor(m + n, dtype=torch.long),
    #     n_cons=torch.tensor(m, dtype=torch.long),
    #     n_vars=torch.tensor(n, dtype=torch.long),
    # )
    
    res = BipartiteHeteroData()
    res["left"].x = const_f
    res["right"].x = var_f
    res["left", "to", "right"].edge_index = edge_index
    res["left", "to", "right"].edge_attr = edge_attr
    return res
#<

def milp_to_vec(problem: LpProblem) -> Tuple[np.ndarray, List[str], np.ndarray]:
    variables = list(problem.variables())
    constraints = list(problem.constraints.values())
    
    n_vars = len(variables)
    n_constraints = len(constraints)
    
    A = np.zeros((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    operators = []
    
    var_to_idx = {var.name: idx for idx, var in enumerate(variables)}
    
    for i, (_, constraint) in enumerate(problem.constraints.items()):
        expr = constraint.toDict()
        for coef in expr['coefficients']:
            var_idx = var_to_idx[coef['name']]
            A[i, var_idx] = coef['value']
        
        b[i] = - expr.get('constant')
        
        if constraint.sense == LpConstraintLE:
            operators.append(-1)
        elif constraint.sense == LpConstraintGE:
            operators.append(1)
        else:
            operators.append(0)
            
    return A, b, operators


def milp_to_graph(problem: LpProblem, add_random_features, random_features_p=None, n_random_features=None, rng=None)-> Data:
    """
    Convert a PuLP problem to graph representation
    """
    A, b, operators= milp_to_vec(problem)
    return vec_to_graph(A, b, operators, add_random_features, random_features_p, n_random_features, rng)


# ----------------------------------------------------------
def vec_to_graph__old(
    A: np.ndarray,                     # Constraint matrix (m x n)
    b: np.ndarray,                     # RHS vector (m)
    operators: List[str],       # List of constraint types ('<=', '=', '>=')
    add_random_features,
    n_random_features=1,
    c: np.ndarray = None,             # Optional objective coefficients (n)
    l: np.ndarray = None,             # Optional lower bounds (n)
    u: np.ndarray = None,             # Optional upper bounds (n)
    I: List[int] = None,               # Optional indices of integer variables,
    **additional_graph_attrs
) -> Data:
    """
    Convert a MILP to graph representation following Section 2.1 of the paper.
    """
    m, n = A.shape  # m constraints, n variables
    
    # Initialize default values if not provided
    if c is None:
        c = np.zeros(n)
    if l is None:
        l = np.zeros(n)  # For binary problems, lower bound is 0
    if u is None:
        u = np.ones(n)   # For binary problems, upper bound is 1
    if I is None:
        I = list(range(n))  # For binary problems, all variables are integer
        
        
    # generate two randome vectors of size n and m
    # rand_vec_n = np.random.rand(n)
    # rand_vec_m = np.random.rand(m)
    
    # 1. Create node features
    # For constraint nodes: pad (bi, ◦i) to match dimension of variable nodes
    # constraint_type_map = {'<=': -1, '=': 0, '>=': 1}
    # constraint_type_values = [constraint_type_map[ct] for ct in operators]
    V_features = np.zeros((m, 4))  # Pad to match W_features dimension
    # V_features = np.zeros((m, 5))  # Pad to match W_features dimension with rand feature
    V_features[:, 0] = b
    # V_features[:, 1] = constraint_type_values
    V_features[:, 1] = operators
    
    # For variable nodes: (cj, lj, uj, τj)
    tau = np.zeros(n)
    tau[I] = 1
    W_features = np.column_stack([c, l, u, tau])
    # W_features = np.column_stack([c, l, u, tau, rand_vec_n]) # add rand feature
    
    # Combine all node features
    x = np.vstack([V_features, W_features])
    
    # 2. Create edges
    edge_index = [[], []]
    edge_attr = []
    
    for i in range(m):
        for j in range(n):
            if A[i, j] != 0:
                # Constraint to variable
                edge_index[0].append(i)
                # edge_index[1].append(j + m) # FIXME: why (+ m)
                edge_index[1].append(j)
                
                edge_attr.append(A[i, j])
                
                #TODO: this is duplication of constraint to variable, also to get work with GCNPolicy as it expects edge from variable to constraint (see left_to_right in GCNPolicy)
                # Variable to constraint
                # edge_index[0].append(j + m)
                # edge_index[1].append(i)
                # edge_attr.append(A[i, j])
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(x)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(edge_attr)
    const_f = torch.FloatTensor(V_features)
    var_f = torch.FloatTensor(W_features)
    
    res = LogicData(
        x=x,
        const_f = const_f,
        var_f = var_f,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=m + n,
        num_constraints=m,
        num_variables=n
    )
    
    # print("Data(.):")
    # print(res)
    # print("data.to_dict")
    # d = res.to_dict()
    # print(d)
    # print("data.from_dict")
    # print(Data.from_dict(d))

    # print(res)
    return res
