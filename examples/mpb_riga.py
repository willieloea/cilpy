# examples/mpb_riga.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem.mpb import MovingPeaksBenchmark
from cilpy.solver.riga import RIGASolver
from cilpy.runner import Runner

if __name__ == '__main__':
    # --- Configure the problem instance ---
    my_problem = MovingPeaksBenchmark(
        dimension=5,
        change_frequency=500
    )

    MAX_ITERATIONS = 5000

    # --- Configure the solver ---
    solver_params = {
        'population_size': 50,
        'p_crossover': 0.9,
        'p_mutation': 0.01,
        'p_immigrants': 0.1,
        'tournament_size': 3,
    }

    # --- Configure and run the experiment ---
    runner = Runner(
        problem=my_problem,
        solver_class=RIGASolver,
        solver_params=solver_params,
        max_iterations=MAX_ITERATIONS,
        change_frequency=my_problem._change_frequency,
        output_filepath="mpb_riga.out.csv"
    )

    runner.run()