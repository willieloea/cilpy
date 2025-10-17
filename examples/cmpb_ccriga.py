# examples/cmpb_ccriga.py

# This block allows running the script from the project's root directory
# without having to install the `cilpy` package.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# --- Import cilpy components ---
from cilpy.runner import ExperimentRunner
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark
from cilpy.solver.solvers.ccriga import CCRIGA


def main():
    """
    An example of using the ExperimentRunner to run CCPSO on the
    Constrained Moving Peaks Benchmark (CMPB).
    """
    # 1. Define the Problem: ConstrainedMovingPeaksBenchmark
    # The problem is composed of two dynamic landscapes.

    # Common parameters
    dimension = 2
    min_coord, max_coord = 0.0, 100.0

    # Parameters for the objective landscape 'f'
    # Changes every 1000 fitness evaluations.
    f_params = {
        "dimension": dimension,
        "num_peaks": 10,
        "min_coord": min_coord,
        "max_coord": max_coord,
        "change_frequency": 100,
        "change_severity": 1.0,
        "height_severity": 7.0,
    }

    # Parameters for the constraint landscape 'g'
    # Changes less frequently but more severely.
    g_params = {
        "dimension": dimension,
        "num_peaks": 5,
        "min_coord": min_coord,
        "max_coord": max_coord,
        "change_frequency": 250,
        "change_severity": 1.5,
        "max_width": 20.0,
        "height_severity": 10.0,
    }

    problem = ConstrainedMovingPeaksBenchmark(
        f_params=f_params,
        g_params=g_params,
        name="CMPB-d5-f1k-g2.5k"
    )

    # 2. Define the Solver and its parameters
    solver_class = CCRIGA
    solver_params = {
        "population_size_x": 30,
        "population_size_l": 30,
        "p_crossover": 0.9,
        "p_mutation": 0.05,
        "p_immigrants": 0.1,
        "tournament_size": 3,
        "lambda_bounds": (0.0, 1000.0),
    }

    # 3. Define the Experiment parameters
    experiment_params = {
        "num_runs": 5,
        "max_iterations": 1000,
        "output_file": "examples/cmpb_ccriga.out.csv",
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
