# run_mpb_experiment.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import mpb
from cilpy.solver import pso
from cilpy.runner import Runner

if __name__ == '__main__':
    # --- Configure the MPB problem instance ---
    # This setup corresponds to a Type III problem (peaks move and change height)
    # with random movement (lambda = 0.0)
    mpb_problem = mpb.MovingPeaksBenchmark(
        dimension=5,
        num_peaks=10,
        change_frequency=100,  # Environment changes every 100 iterations
        height_severity=7.0,   # Peaks change height significantly
        change_severity=1.0,   # Peaks move
        lambda_param=0.0       # Movement is uncorrelated (random)
    )

    solver_params = {
        'population_size': 40,
    }

    runner = Runner(
        problem=mpb_problem,
        solver_class=pso.GbestPSO,
        solver_params=solver_params,
        max_iterations=5000,
        change_frequency=mpb_problem._change_frequency, # Pass freq to runner
        output_filepath="mpb_results.csv"
    )

    runner.run()