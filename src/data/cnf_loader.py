# cnf_loader.py
from pathlib import Path 
import torch
from dataclasses import dataclass
from typing import List, Tuple
import random
from tqdm import tqdm
from milp.utils.cnf import CNFProcessor

@dataclass
class CNFDatasetInstance:
   cons_features: torch.Tensor
   edge_indices: torch.Tensor
   edge_features: torch.Tensor
   var_features: torch.Tensor
   n_cons: torch.Tensor
   n_vars: torch.Tensor
   label: torch.Tensor

class CNFLoader:
   @staticmethod
   def _resolve_path(data_dir: str) -> Tuple[Path, Path, Path]:
       data_dir = Path(data_dir)
       sat_dir = data_dir / 'sat'
       unsat_dir = data_dir / 'unsat'
       
       if not sat_dir.exists() or not unsat_dir.exists():
           raise ValueError(f"Both 'sat' and 'unsat' directories must exist in {data_dir}")
           
       return data_dir, sat_dir, unsat_dir
       
   @staticmethod
   def _load_cnf_files(directory: Path) -> List[str]:
       return [str(f) for f in directory.glob("*.cnf")]

   @staticmethod 
   def _process_single_cnf(filepath: str, is_sat: bool) -> CNFDatasetInstance:
       try:
           processor = CNFProcessor()
           formula = processor.load_dimacs(filepath)
           
           return CNFDatasetInstance(
               cons_features=torch.FloatTensor(formula.clause_features),
               edge_indices=torch.LongTensor(formula.edge_indices),
               edge_features=torch.FloatTensor(formula.edge_features),
               var_features=torch.FloatTensor(formula.var_features),
               n_cons=torch.tensor(formula.num_clauses, dtype=torch.long),
               n_vars=torch.tensor(formula.num_vars, dtype=torch.long),
               label=torch.FloatTensor([1 if is_sat else 0])
           )
           
       except Exception as e:
           print(f"Error processing {filepath}: {str(e)}")
           return None

   @staticmethod
   def load_dataset(data_dir, max_samples: int = None, split_ratio: float = 0.8) -> Tuple[List, List, List, List]:
       _, sat_dir, unsat_dir = CNFLoader._resolve_path(data_dir)
       
       sat_files = CNFLoader._load_cnf_files(sat_dir)
       unsat_files = CNFLoader._load_cnf_files(unsat_dir)
       
       n_samples = min(len(sat_files), len(unsat_files))
       if max_samples:
           n_samples = min(n_samples, max_samples)
       
       sat_files = random.sample(sat_files, n_samples)
       unsat_files = random.sample(unsat_files, n_samples)
       
       print(f"Loading {n_samples} SAT and {n_samples} UNSAT instances...")
       
       all_data = []
       all_labels = []
       
       for filepath in tqdm(sat_files, desc="Processing SAT instances"):
           data = CNFLoader._process_single_cnf(filepath, is_sat=True)
           if data is not None:
               all_data.append(data)
               all_labels.append(data.label)
               
       for filepath in tqdm(unsat_files, desc="Processing UNSAT instances"):
           data = CNFLoader._process_single_cnf(filepath, is_sat=False)
           if data is not None:
               all_data.append(data)
               all_labels.append(data.label)
               
       combined = list(zip(all_data, all_labels))
       random.shuffle(combined)
       all_data, all_labels = zip(*combined)
       
       split_idx = int(len(all_data) * split_ratio)
       
       train_data = all_data[:split_idx]
       test_data = all_data[split_idx:]
       train_labels = all_labels[:split_idx] 
       test_labels = all_labels[split_idx:]
       
       train_dataset = [(d.cons_features, d.edge_indices, d.edge_features, d.var_features) 
                       for d in train_data]
       
       test_dataset = [(d.cons_features, d.edge_indices, d.edge_features, d.var_features)
                      for d in test_data]
       
       print(f"\nDataset statistics:")
       print(f"Total samples: {len(all_data)}")
       print(f"Training samples: {len(train_dataset)}")
       print(f"Testing samples: {len(test_dataset)}")
       
       return train_dataset, train_labels, test_dataset, test_labels
