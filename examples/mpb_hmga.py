# examples/mpb_hmga.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem.mpb import MovingPeaksBenchmark
from solvers.hmga import HyperMutationGA
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
        'crossover_prob': 0.9,
        'pm': 0.01,             # Standard mutation rate
        'phyper': 0.2,          # Hyper-mutation rate
        'hyper_total': 5,       # Generations in hyper-mutation mode
        'tournament_size': 3,
    }

    # --- Configure and run the experiment ---
    runner = Runner(
        problem=my_problem,
        solver_class=HyperMutationGA,
        solver_params=solver_params,
        max_iterations=MAX_ITERATIONS,
        change_frequency=my_problem._change_frequency, # Pass freq to runner
        output_filepath="mpb_hmga.out.csv"
    )

    runner.run()