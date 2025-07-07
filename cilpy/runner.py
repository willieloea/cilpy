# cilpy/runner.py

from typing import Type, Dict, Any
import csv

from .problem import Problem
from .solver import Solver

class Runner():
    """
    Used for running optimization experiments and logging results
    """
    def __init__(self,
                problem: Problem,
                solver_class: Type[Solver],
                solver_params: Dict[str, Any],
                max_iterations: int=500,
                output_filepath: str="output.csv",
                change_frequency: int=0):
        """
        Initializes the experiment runner.

        Args:
            problem: An instance of a class that implements the Problem
            interface
            solver_class: The class of the solver to use
            solver_params: A dictionary of parameters for the solver constructor
            max_iterations: The total number of iterations to run
            change_frequency: How often to call problem.change_environment(), 0
                              for a static environment
        """
        self.problem = problem
        self.solver = solver_class(problem=problem, **solver_params)
        self.max_iterations = max_iterations
        self.output_filepath = output_filepath
        self.change_frequency = change_frequency
        self.results = []

    def run(self) -> None:
        """
        Executes the full experiment.
        """
        print(f"--- Starting Experiment: {self.solver.__class__.__name__} on \
{self.problem.name} ---")
        print(f"Saving results to: {self.output_filepath}")

        # Write header to the results list
        self.results.append(['iteration','best_fitness'])

        for i in range(self.max_iterations):
            # Check if the environment should change before the solver's step
            is_objective_dynamic, _ = self.problem.is_dynamic()
            if is_objective_dynamic and self.change_frequency > 0:
                self.problem.change_environment(i)

            # Advance the solver by one step
            self.solver.step()

            # Get data logging - best result, in this case
            _, best_fitness_list = self.solver.get_best()
            # Assuming single objective, get the first value
            # best_fitness = best_fitness_list[0]
            self.results.append([i, best_fitness_list])

            # (optional) Log fitness to console
            if (i + 1) % 50 == 0:
                print(f"  Iteration {i+1}/{self.max_iterations} complete. \
    Current Best Fitness: {best_fitness_list[0]}")
    # Current Best Fitness: {best_fitness:.4f}")

        # Save results to CSV
        self._save_to_csv()

    def _save_to_csv(self) -> None:
        with open(self.output_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.results)
        print("Results successfully saved.")
