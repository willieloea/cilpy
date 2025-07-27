# examples/cmpb_pso.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import cmpb
from cilpy.runner import Runner
from cilpy.solver.solvers import pso
from cilpy.solver.chm.debs_rules import DebsRules
from cilpy.solver.chm.penalty import StaticPenalty

if __name__ == '__main__':
    # --- Configure the CMPB problem instance ---

    # Parameters for the objective landscape (f)
    f_params = {
        'dimension': 1,
        'num_peaks': 10,
        'change_frequency': 0,
        'height_severity': 5.0,
        'width_severity': 0.5,
        'change_severity': 1.0,
        'lambda_param': 0.1,
        'problem_name': 'ObjectiveLandscape'
    }

    # Parameters for the constraint landscape (g)
    g_params = {
        'dimension': 1,
        'num_peaks': 15,
        'change_frequency': 0,
        'height_severity': 10.0,
        'width_severity': 1.0,
        'change_severity': 1.5,
        'lambda_param': 0.0,
        'problem_name': 'ConstraintLandscape'
    }

    # Create the composed CMPB problem
    cmpb_problem = cmpb.ConstrainedMovingPeaksBenchmark(
        f_params=f_params,
        g_params=g_params,
        problem_name="DynamicConstrainedProblem"
    )
    
    # --- Configure the solver and runner ---
    
    solver_params = {
        'population_size': 30,
    }
    
    # The runner needs to know how often to trigger a change.
    # We can use the more frequent change from the g_landscape.
    change_freq = g_params['change_frequency']
    
    # --- Experiment 1: Using Deb's Rules ---
    print("Running PSO with Deb's Rules...")
    debs_handler = DebsRules(problem=cmpb_problem)
    runner_debs = Runner(
        problem=cmpb_problem,
        solver_class=pso.GbestPSO,
        solver_params={
            'population_size': 30,
            'constraint_handler': debs_handler # Pass the CHM instance
        },
        max_iterations=5000,
        output_filepath="cmpb_pso_debs.out.csv"
    )
    runner_debs.run()

    # --- Experiment 2: Using Static Penalty ---
    print("\nRunning PSO with Static Penalty...")
    penalty_handler = StaticPenalty(problem=cmpb_problem, penalty_coefficient=1e7)
    runner_penalty = Runner(
        problem=cmpb_problem,
        solver_class=pso.GbestPSO,
        solver_params={
            'population_size': 30,
            'constraint_handler': penalty_handler
        },
        max_iterations=5000,
        output_filepath="cmpb_pso_penalty.out.csv"
    )
    runner_penalty.run()
