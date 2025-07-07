from cilpy.problem import sphere
from cilpy.solver import pso
from cilpy.runner import Runner

if __name__ == '__main__':
    my_problem = sphere.Sphere(dimension=10)

    solver_params = {
        'population_size': 40,
    }

    runner = Runner(
        problem=my_problem,
        solver_class=pso.GbestPSO,
        solver_params=solver_params,
    )

    runner.run()
