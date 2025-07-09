# examples/mpb_gvpso.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import mpb
from cilpy.solver import gvpso
from cilpy.runner import Runner

if __name__ == '__main__':
    # --- Configure the problem instance ---
    mpb_problem = mpb.MovingPeaksBenchmark(
        dimension=10,
        num_peaks=10,
        change_frequency=250,
        height_severity=7.0,
        change_severity=1.0,
        lambda_param=0.0,
        min_coord=-50.0,
        max_coord=50.0
    )

    MAX_ITERATIONS = 5000

    # --- Configure the solver ---
    solver_params = {
        'swarm_size': 40,
        'e': 0.3, # Exploitation probability
    }

    # --- Configure and run the experiment ---
    runner = Runner(
        problem=mpb_problem,
        solver_class=gvpso.GVPSOSolver,
        solver_params=solver_params,
        max_iterations=MAX_ITERATIONS,
        change_frequency=mpb_problem._change_frequency,
        output_filepath="mpb_gvpso.out.csv"
    )

    runner.run()