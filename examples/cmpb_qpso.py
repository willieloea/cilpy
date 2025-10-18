# examples/cmpb_qpso.py

# This block allows running the script from the project's root directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# --- Import cilpy components ---
from cilpy.old_runner import ExperimentRunner
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark
from cilpy.solver.solvers.qpso import QPSO


def main():
    """
    An example of using the ExperimentRunner to run QPSO on the
    Constrained Moving Peaks Benchmark (CMPB).
    """
    # 1. Define the Problem: ConstrainedMovingPeaksBenchmark
    dimension = 2
    min_coord, max_coord = 0.0, 100.0

    # Parameters for the objective landscape 'f'
    f_params = {
        "dimension": dimension,
        "num_peaks": 10,
        "min_coord": min_coord,
        "max_coord": max_coord,
        "change_frequency": 100,
        "change_severity": 1.0,
    }

    # Parameters for the constraint landscape 'g'
    g_params = {
        "dimension": dimension,
        "num_peaks": 5,
        "min_coord": min_coord,
        "max_coord": max_coord,
        "change_frequency": 250,
        "change_severity": 1.5,
    }

    problem = ConstrainedMovingPeaksBenchmark(
        f_params=f_params,
        g_params=g_params,
        name="CMPB_f1000_g2500"
    )

    # 2. Define the Experiment parameters
    # This must be defined before the solver parameters, as QPSO needs it.
    experiment_params = {
        "num_runs": 5,
        "max_iterations": 1000,
        "output_file": "examples/cmpb_qpso.out.csv",
    }

    # 3. Define the Solver and its parameters
    solver_class = QPSO
    solver_params = {
        "swarm_size": 50,
        "alpha_start": 1.0,
        "alpha_end": 0.5,
        "distribution": "gaussian",  # Or "uniform"
        # Pass max_iterations to the solver for alpha scheduling
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