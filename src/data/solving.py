from pulp import *
import time
import os
from pysat.formula import CNF
from pysat.solvers import Solver
from collections import defaultdict

def solve_SAT_problem2(batch):
    # keys = ["feasibility", "time", "variable_values"]
    output = defaultdict(list)

    for folder, filename in zip(batch['folder'], batch['filename']):
        filename = os.path.join(folder, filename)
        cnf = CNF(from_file=filename)
        with Solver(name='g3', use_timer=True) as solver:
            # Add the clauses from the CNF formula to the solver
            solver.append_formula(cnf)
            sat = solver.solve()
            output['label'].append(1 if sat else 0)
            output['variable_values'].append(" ".join(map(str, solver.get_model())) if sat else "")
            output['time'].append(solver.time())
    return output
#<


def solve_SAT_problem(cnf_file):
    """
    Solve a SAT problem and return the solution
    
    Args:
        cnf_file: Path to the DIMACS CNF file to solve
    
    Returns:
        dict: A dictionary mapping problem names to solution dictionaries containing:
            - feasibility (bool): Whether solution was found
            - variable_values (list): Values for each variable (if feasible)
            - time (float): Time taken to solve
    """
    # Initialize the result dictionary
    result = {
        'feasibility': False,
        'variable_values': None,
        'time': None
    }
    
    # Load the CNF formula from the file
    cnf = CNF(from_file=cnf_file)
    
    # Create a Glucose3 solver instance
    start_time = time.time()
    with Solver(name='g3') as solver:
        # Add the clauses from the CNF formula to the solver
        solver.append_formula(cnf)

        # Check if the formula is satisfiable
        if solver.solve():
            result['feasibility'] = True
            # Get a satisfying assignment
            result['variable_values'] = solver.get_model()
        else:
            result['feasibility'] = False
        
        # Record the time taken to solve
        result['time'] = time.time() - start_time
    
    return result
