# cilpy/runner.py

import csv
from pathlib import Path
from typing import Type, Dict, Any
import time

from .problem import Problem
from .solver import Solver


class ExperimentRunner:
    """
    A generic runner to execute optimization experiments.

    This class orchestrates the execution of a given solver on a given problem
    for a specified number of runs and iterations, logging the results to a
    CSV file. It is designed to be generic, relying on the `Problem` and
    `Solver` interfaces.
    """

    def __init__(
        self,
        problem: Problem,
        solver_class: Type[Solver],
        solver_params: Dict[str, Any],
        num_runs: int,
        max_iterations: int,
        output_file: str,
    ):
        """
        Initializes the ExperimentRunner.

        Args:
            problem (Problem): An instance of a class that implements the
                Problem interface.
            solver_class (Type[Solver]): The class of the solver to be used
                (e.g., GbestPSO), not an instance.
            solver_params (Dict[str, Any]): A dictionary of parameters to be
                passed to the solver's constructor (e.g., {'swarm_size': 30}).
            num_runs (int): The number of independent runs to perform.
            max_iterations (int): The number of iterations (steps) per run.
            output_file (str): The path to the CSV file where results will be
                saved.
        """
        if not issubclass(solver_class, Solver):
            raise TypeError("solver_class must be a subclass of cilpy.solver.Solver")
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of a cilpy.problem.Problem subclass")

        self.problem = problem
        self.solver_class = solver_class
        self.solver_params = solver_params
        self.num_runs = num_runs
        self.max_iterations = max_iterations
        self.output_file = Path(output_file)

    def run(self) -> None:
        """
        Executes the full experiment and saves the results.

        This method iterates through the specified number of runs. For each run,
        it creates a new solver instance and runs it for the specified number
        of iterations. The best fitness found at each iteration is logged.
        """
        print(f"Starting experiment: {self.solver_class.__name__} on {self.problem.name}")
        print(f"Configuration: {self.num_runs} runs, {self.max_iterations} iterations/run.")
        print(f"Results will be saved to: {self.output_file}")

        # Ensure the output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        header = ["run", "iteration", "best_fitness", "best_solution"]

        with self.output_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            total_start_time = time.time()

            for run_id in range(1, self.num_runs + 1):
                run_start_time = time.time()
                print(f"--- Starting Run {run_id}/{self.num_runs} ---")

                # Re-instantiate the solver for each run to ensure independence
                solver = self.solver_class(self.problem, **self.solver_params)

                for iteration in range(1, self.max_iterations + 1):
                    solver.step()
                    best_solution, best_fitness = solver.get_result()

                    # Log data for this iteration
                    # The solution is converted to a string for generic CSV storage
                    solution_str = str(best_solution).replace('\n', '')
                    writer.writerow([run_id, iteration, best_fitness, solution_str])

                run_end_time = time.time()
                best_solution, best_fitness = solver.get_result()
                print(
                    f"Run {run_id} finished in {run_end_time - run_start_time:.2f}s. "
                    f"Best fitness: {best_fitness:.4e}"
                )

        total_end_time = time.time()
        print("\n--- Experiment Finished ---")
        print(f"Total execution time: {total_end_time - total_start_time:.2f}s")