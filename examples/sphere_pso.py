# sphere_pso.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cilpy.problem import sphere
from cilpy.solver import pso
from cilpy.runner import Runner

if __name__ == '__main__':
    my_problem = sphere.Sphere(dimension=100)

    solver_params = {
        'population_size': 30,
    }

    runner = Runner(
        problem=my_problem,
        solver_class=pso.GbestPSO,
        solver_params=solver_params,
        max_iterations=10000,
        output_filepath="sphere_pso_results.csv"
    )

    runner.run()
