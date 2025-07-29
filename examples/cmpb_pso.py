# examples/cmpb_pso.py

import numpy as np

# This block allows running the script from the project's root directory
# without having to install the `cilpy` package.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# --- Import cilpy components ---
from cilpy.runner import ExperimentRunner
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark
from cilpy.solver.solvers.pso import GbestPSO
# DebsRules is the default CHM in GbestPSO, so explicit import isn't strictly
# needed but is good for clarity if you want to swap it out.
# from cilpy.solver.chm.debs_rules import DebsRules


def main():
    """
    An example of using the ExperimentRunner to run GbestPSO on the
    Constrained Moving Peaks Benchmark (CMPB).
    """
    # 1. Define the Problem: ConstrainedMovingPeaksBenchmark
    # CMPB is composed of two Moving Peaks Benchmark instances. We define
    # their parameters separately.

    # Common parameters for both landscapes
    dimension = 2
    min_coord, max_coord = 0.0, 100.0

    # Parameters for the objective landscape 'f'
    # Let's make this one change more frequently.
    f_params = {
        "dimension": dimension,
        "num_peaks": 10,
        "min_coord": min_coord,
        "max_coord": max_coord,
        "change_frequency": 100,  # Changes every 1000 evaluations
        "change_severity": 1.0,
    }

    # Parameters for the constraint landscape 'g'
    # Let's make this one change less frequently but more severely.
    g_params = {
        "dimension": dimension,
        "num_peaks": 5,
        "min_coord": min_coord,
        "max_coord": max_coord,
        "change_frequency": 250,  # Changes every 2500 evaluations
        "change_severity": 1.5,
        "max_width": 15.0,
    }

    problem = ConstrainedMovingPeaksBenchmark(
        f_params=f_params,
        g_params=g_params,
        name="CMPB_f1000_g2500"
    )

    # 2. Define the Solver and its parameters
    solver_class = GbestPSO
    solver_params = {
        "swarm_size": 30,
        "w": 0.7298,
        "c1": 1.49618,
        "c2": 1.49618,
        # GbestPSO will default to DebsRules if constraint_handler is None,
        # which is the desired behavior for this constrained problem.
        "constraint_handler": None,
    }

    # 3. Define the Experiment parameters
    # A longer run is needed to observe the effects of the dynamic environment.
    experiment_params = {
        "num_runs": 5,
        "max_iterations": 1000,
        "output_file": "examples/cmpb_pso.out.csv",
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