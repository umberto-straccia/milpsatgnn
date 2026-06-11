"""
Module containing utility to maps between DIMACS and MPS formats.
"""

from typing import List, Tuple
import pulp
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional

def parse_dimacs(dimacs_file: str) -> Tuple[List[List[int]], List[int]]:
    """
    Parse DIMACS CNF file.
    
    Args:
        dimacs_file (str): Path to DIMACS CNF file
        
    Returns:
        Tuple[List[List[int]], List[int]]: Tuple containing:
            - List of clauses where each clause is a list of literals
            - Sorted list of variables
            
    Example:
        >>> clauses, variables = parse_dimacs("example.dimacs")
    """
    clauses = []
    variables = set()
    
    with open(dimacs_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('c') or line.startswith('p'):
                continue
            literals = list(map(int, line.split()))
            if literals[-1] == 0:
                literals = literals[:-1]
            clauses.append(literals)
            variables.update(abs(lit) for lit in literals)
    
    return clauses, sorted(variables)



# converters.py
"""
Module containing conversion functionality between DIMACS and MPS formats.
"""


def get_automatic_output_path(input_path: str) -> str:
    """
    Generate the automatic output path for an MPS file based on input DIMACS file path.
    
    Args:
        input_path (str): Path to the input DIMACS file
        
    Returns:
        str: Generated output path for the MPS file
        
    Examples:
        >>> get_automatic_output_path("problem.dimacs")
        'problem.mps'
        >>> get_automatic_output_path("path/to/problem.dimacs")
        'path/to/problem.mps'
        >>> get_automatic_output_path("path/to/problem.cnf")
        'path/to/problem.mps'
        >>> get_automatic_output_path("/absolute/path/problem.txt")
        '/absolute/path/problem.mps'
    """
    path = Path(input_path)
    return str(path.with_suffix('.mps'))

def convert_dimacs(dimacs_file: str, 
                  output_path: Optional[str] = None,
                  save_to_file: bool = True) -> Union[pulp.LpProblem, Tuple[pulp.LpProblem, str]]:
    """
    Convert a DIMACS file to MPS format with flexible output handling.
    This function combines parsing, conversion, and optional file saving in one step.
    
    Args:
        dimacs_file (str): Path to the input DIMACS file
        output_path (Optional[str]): Path where the MPS file should be saved. 
                                   If None and save_to_file is True, 
                                   automatically generates path by replacing the input file's
                                   extension with .mps in the same directory.
        save_to_file (bool): Whether to save the result to an MPS file
        
    Returns:
        Union[pulp.LpProblem, Tuple[pulp.LpProblem, str]]:
            - If save_to_file is False: Returns only the MILP problem
            - If save_to_file is True: Returns tuple of (MILP problem, output file path)
            
    Raises:
        FileNotFoundError: If the input DIMACS file doesn't exist
        ValueError: If output_path is invalid or write permission is denied
        
    Examples:
        >>> # Convert and save in same directory (input: /path/to/prob.dimacs -> output: /path/to/prob.mps)
        >>> prob, out_path = convert_dimacs("/path/to/prob.dimacs")
        >>> print(out_path)
        '/path/to/prob.mps'
        
        >>> # Convert and save with custom output path
        >>> prob, out_path = convert_dimacs("prob.dimacs", output_path="output/sol.mps")
        >>> print(out_path)
        'output/sol.mps'
        
        >>> # Just get the MILP problem without saving
        >>> prob = convert_dimacs("prob.dimacs", save_to_file=False)
    """
    # Validate input file
    if not os.path.exists(dimacs_file):
        raise FileNotFoundError(f"Input file not found: {dimacs_file}")
    
    # Parse DIMACS file
    try:
        clauses, variables = parse_dimacs(dimacs_file)
    except Exception as e:
        raise ValueError(f"Error parsing DIMACS file: {str(e)}")
    
    # Convert to MILP
    milp_problem = dimacs_to_milp(clauses, variables)
    
    if not save_to_file:
        return milp_problem
    
    # Determine output path
    if output_path is None:
        output_path = get_automatic_output_path(dimacs_file)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write to MPS file
    try:
        write_mps(milp_problem, output_path)
    except Exception as e:
        raise ValueError(f"Error writing MPS file: {str(e)}")
    
    return milp_problem, output_path

def dimacs_to_milp(clauses: List[List[int]], variables: List[int]) -> pulp.LpProblem:
    """
    Convert DIMACS CNF format to MILP problem.
    
    Args:
        clauses (List[List[int]]): List of clauses where each clause is a list of literals
        variables (List[int]): List of variables in the problem
        
    Returns:
        pulp.LpProblem: MILP problem instance
        
    Example:
        >>> clauses = [[1, -2, 3], [-1, 2, -3]]
        >>> variables = [1, 2, 3]
        >>> prob = dimacs_to_milp(clauses, variables)
    """
    # Create MILP problem instance
    model = pulp.LpProblem("MILP_from_DIMACS", pulp.LpMinimize)
    
    # Create binary decision variables
    lp_vars = {var: pulp.LpVariable(f"x_{var}", cat=pulp.const.LpBinary) for var in variables}

    # Add trivial objective function (satisfiability only)
    model += 0
    # Add constraints for each clause
    for clause in clauses:
        # model += pulp.lpSum(lp_vars[abs(lit)] if lit > 0 else (1 - lp_vars[abs(lit)]) for lit in clause) >= 1
        model += pulp.lpSum(lp_vars[abs(lit)] if lit >=0 else -lp_vars[abs(lit)] for lit in clause) >= 1- len(list(filter(lambda x: x < 0, clause)))
    
    return model

def write_mps(problem: pulp.LpProblem, output_file: str) -> None:
    """
    Write MILP problem to MPS format file.
    
    Args:
        problem (pulp.LpProblem): MILP problem instance
        output_file (str): Path to output MPS file
        
    Example:
        >>> write_mps(prob, "output.mps")
    """
    problem.writeMPS(output_file)