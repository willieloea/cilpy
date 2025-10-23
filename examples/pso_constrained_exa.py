# example/pso_constrained_exa.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.constrained import G01
from cilpy.solver.pso import PSO

# --- 1. Define the Problem ---
problems_to_run = [
    G01(),
]

# --- 2. Define the Solver ---
# NOTE: At the time of writing, the PSO does not handle constraints.
solver_configs = [
    {
        "class": PSO,
        "params": {
            "name": "PSO_for_Constrained",
            "swarm_size": 50,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 10
max_iter = 2000

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
