# examples/mpb_pso.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import mpb
from solvers import pso
# from cilpy.compare import test1, test2, ... XXX TODO XXX
from cilpy.runner import Runner

if __name__ == '__main__':
    # --- Configure the MPB problem instance ---
    mpb_problem = mpb.MovingPeaksBenchmark(
        dimension=2,
        num_peaks=10,
        change_frequency=100,  # Environment changes every 100 iterations
        height_severity=7.0,   # Peaks change height significantly
        change_severity=1.0,   # Peaks move
        lambda_param=0.0       # Movement is uncorrelated (random)
    )

    MAX_ITERATIONS = 5000

    # --- Configure the PSO solver ---
    solver_params = {
        'population_size': 30,
        'max_iterations': MAX_ITERATIONS
    }

    # --- Configure comparisons to run XXX ---
    # compare_params = { TODO }

    # --- Configure and run the experiment ---
    runner = Runner(
        problem=mpb_problem,
        solver_class=pso.GbestPSO,
        solver_params=solver_params,
        max_iterations=MAX_ITERATIONS,
        change_frequency=mpb_problem._change_frequency, # Pass freq to runner
        output_filepath="mpb_pso.out.csv"
        # compare_params
    )

    runner.run()