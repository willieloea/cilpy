# run_cmpb_experiment.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import cmpb
from cilpy.solver import pso
from cilpy.runner import Runner

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
        'population_size': 50,
    }
    
    # The runner needs to know how often to trigger a change.
    # We can use the more frequent change from the g_landscape.
    change_freq = g_params['change_frequency']
    
    runner = Runner(
        problem=cmpb_problem,
        solver_class=pso.GbestPSO,
        solver_params=solver_params,
        max_iterations=10000,
        change_frequency=change_freq,
        output_filepath="cmpb_results.csv"
    )

    runner.run()