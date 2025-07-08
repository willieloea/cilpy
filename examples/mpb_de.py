# examples/mpb_de.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import mpb
from cilpy.solver import de_rand_1_bin
from cilpy.runner import Runner

if __name__ == '__main__':
    # --- Configure the MPB problem instance ---
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
    
    MAX_ITERATIONS = 10000

    # --- Configure the DE solver ---
    solver_params = {
        'population_size': 100,      # DE often benefits from larger populations
        'scale_factor': 0.8,
        'crossover_prob': 0.9,
    }

    # --- Configure and run the experiment ---
    runner = Runner(
        problem=mpb_problem,
        solver_class=de_rand_1_bin.DifferentialEvolutionSolver,
        solver_params=solver_params,
        max_iterations=MAX_ITERATIONS,
        change_frequency=mpb_problem._change_frequency,
        output_filepath="mpb_de_out.csv"
    )

    runner.run()