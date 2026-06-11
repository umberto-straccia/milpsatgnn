# MILP-SAT-GNN: Yet Another Neural SAT solver

## Set-up
- The list of the required packages is in the file: requirements, in the root folder;
- The Python environment has been created with uv: https://github.com/astral-sh/uv
- Using Python 3.10.15

We conducted the experiments on a Linux machines using an nVidia GPU. Please adapt the following instructions to your own specific experimental machine.

Create the environment and install required packages by executing in the root folder of this repository:
- `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `uv venv --python 3.10.15`
- `source .venv/bin/activate`
- `uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0`
- `uv pip install -r requirements.txt`


## Configuration
The configuration file `dataconfig/full-exp.yaml` executes a full experiment.


## Run
- `cd src`
- `python main.py --config ../dataconfig/full-exp.yaml`



## Cite as

@inproceedings{Cardillo26a,  
	author = {Franco Alberto Cardillo, Hamza Khyari, Riccardo Paoli, Umberto Straccia},  
	booktitle = {Proceedings of the 30th International Conference on Knowledge-Based and Intelligent Information \& Engineering Systems (KES-2026)},  
	publisher = {Elsevier},  
	series = {Procedia Computer Science},  
	title = {MILP-SAT-GNN: Yet Another Neural SAT solver},  
	year = {2026}
}








