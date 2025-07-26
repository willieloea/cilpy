# examples/mpb_qpso.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import mpb
from solvers import qpso
from cilpy.runner import Runner

if __name__ == '__main__':
    # --- Configure the MPB problem instance ---
    mpb_problem = mpb.MovingPeaksBenchmark(
        dimension=2,
        num_peaks=10,
        change_frequency=100,
        height_severity=7.0,
        change_severity=1.0,
        lambda_param=0.0
    )
    
    MAX_ITERATIONS = 5000

    # --- Configure the QPSO solver ---
    solver_params = {
        'swarm_size': 30,
        'alpha_start': 1.0,
        'alpha_end': 0.5,
        'max_iterations': MAX_ITERATIONS, # QPSO needs to know this for alpha scheduling
        'distribution': 'gaussian'        # Can be 'uniform' or 'gaussian'
    }

    # --- Configure and run the experiment ---
    runner = Runner(
        problem=mpb_problem,
        solver_class=qpso.QPSOSolver,
        solver_params=solver_params,
        max_iterations=MAX_ITERATIONS,
        change_frequency=mpb_problem._change_frequency,
        output_filepath="mpb_qpso.out.csv"
    )

    runner.run()