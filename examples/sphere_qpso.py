# examples/sphere_qpso.py

import numpy as np

# --- Import cilpy components ---
# Note the relative imports might need adjustment based on how you run the script.
# If cilpy is installed, you can just do `from cilpy.runner import ExperimentRunner`.
from cilpy.old_runner import ExperimentRunner
from cilpy.problem.functions import Sphere
from cilpy.solver.solvers.qpso import QPSO

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

    # 2. Define the Experiment parameters
    # This must be defined before the solver parameters, as QPSO needs it.
    experiment_params = {
        "num_runs": 5,
        "max_iterations": 1000,
        "output_file": "examples/sphere_qpso.out.csv",
    }

    # 3. Define the Solver and its parameters
    solver_class = QPSO
    solver_params = {
        "swarm_size": 50,
        "alpha_start": 1.0,
        "alpha_end": 0.5,
        "distribution": "gaussian",  # Or "uniform"
        # CRITICAL: Pass max_iterations to the solver for alpha scheduling
        "max_iterations": experiment_params["max_iterations"],
        "constraint_handler": None,  # Defaults to DebsRules
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
