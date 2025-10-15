# Quick Start Guide
To use `cilpy`, first download the repository. Then navigate into the root
directory of the repository, and install `cilpy` as a python package by running
```bash
pip install -e .
```
You can then construct a problem that implements the `cilpy/problem` interface,
construct an optimizer/solver that implements the `cilpy/solver` interface,
create a runner instance, and run the runner. Below is a pseudocode example:
```python
from cilpy.problem import some_problem
from cilpy.solver import some_solver
from cilpy.runner import Runner

if __name__ == '__main__':
    # --- Configure the problem instance ---
    my_problem = some_problem.SomeProblemClass(
        dimension=2
        # Other parameters
    )

    MAX_ITERATIONS = 5000

    # --- Configure the solver ---
    solver_params = {
        'population_size': 30,
        'max_iterations': MAX_ITERATIONS,
        # Other parameters
    }

    # --- Configure and run the experiment ---
    runner = Runner(
        problem=my_problem,
        solver_class=some_solver.SomeSolverClass,
        solver_params=solver_params,
        max_iterations=MAX_ITERATIONS,
        change_frequency=my_problem._change_frequency, # Pass freq to runner
        output_filepath="out.csv"
    )

    runner.run()
```
