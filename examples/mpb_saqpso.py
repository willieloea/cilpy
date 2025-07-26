# examples/sphere_saqpso.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import mpb
from solvers.saqpso import SaQPSOSolver
from cilpy.runner import Runner

if __name__ == '__main__':
    # --- Configure the problem instance ---
    mpb_problem = mpb.MovingPeaksBenchmark(
        dimension=2,
        num_peaks=10,
        change_frequency=100,
        height_severity=7.0,
        change_severity=1.0,
        lambda_param=0.0
    )

    MAX_ITERATIONS = 5000

    # --- Configure the solver ---
    solver_params = {
        'swarm_size': 30,
        'neutral_ratio': 0.5, # 50% neutral, 50% quantum
        'w': 0.729,
        'c1': 1.494,
        'c2': 1.494,
    }

    # --- Configure and run the experiment ---
    runner = Runner(
        problem=mpb_problem,
        solver_class=SaQPSOSolver,
        solver_params=solver_params,
        max_iterations=MAX_ITERATIONS,
        output_filepath="mpb_saqpso.out.csv"
    )

    runner.run()