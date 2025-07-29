# examples/sphere_pso.py

import numpy as np

# --- Import cilpy components ---
# Note the relative imports might need adjustment based on how you run the script.
# If cilpy is installed, you can just do `from cilpy.runner import ExperimentRunner`.
from cilpy.runner import ExperimentRunner
from cilpy.problem.functions import Sphere
from cilpy.solver.solvers.pso import GbestPSO

# This block allows running the script from the root directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    """
    An example of using the ExperimentRunner to run PSO on the Sphere function.
    """
    # 1. Define the Problem
    dimension = 2
    bounds = (
        np.array([-5.12] * dimension),
        np.array([5.12] * dimension)
    )
    problem = Sphere(dimension=dimension, bounds=bounds)

    # 2. Define the Solver and its parameters
    solver_class = GbestPSO
    solver_params = {
        "swarm_size": 30,
        "w": 0.7298,
        "c1": 1.49618,
        "c2": 1.49618,
        "constraint_handler": None,  # Sphere is unconstrained
    }

    # 3. Define the Experiment parameters
    experiment_params = {
        "num_runs": 5,
        "max_iterations": 1000,
        "output_file": "examples/sphere_pso.out.csv",
    }

    # 4. Create and run the experiment
    runner = ExperimentRunner(
        problem=problem,
        solver_class=solver_class,
        solver_params=solver_params,
        **experiment_params
    )
    runner.run()


if __name__ == "__main__":
    main()