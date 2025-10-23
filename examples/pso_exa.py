# example/pso_exa.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.unconstrained import Sphere
from cilpy.solver.pso import PSO

# --- 1. Define the Problems ---
dim = 3
dom = (-5.12, 5.12)
problems_to_run = [
    Sphere(dimension=dim, domain=dom)
]

# --- 2. Define the Solvers and their parameters ---
# Note: The 'problem' parameter is omitted here as the runner will assign it.
solver_configs = [
    {
        "class": PSO,
        "params": {
            "name": "PSO",
            "swarm_size": 30,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
            "k": 1,
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 1
max_iter = 10

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
