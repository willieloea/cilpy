# cilpy/runner.py
"""The experiment runner: Orchestrates computational intelligence experiments.

This module provides the `ExperimentRunner` class, which is the primary tool for
setting up, executing, and logging benchmark experiments in a structured and
reproducible way.
"""

import time
import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Sequence

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver


class ExperimentRunner:
    """Orchestrates the execution of computational intelligence experiments.

    This class is the main entry point for running experiments in `cilpy`. It
    automates the process of applying multiple solver configurations to a set of
    problems, handling independent runs, iteration loops, and results logging.

    The runner systematically pairs each solver with each problem, creating a
    dedicated output file for each combination. This declarative approach allows
    users to define complex experiments with minimal boilerplate code.

    Example:
        .. code-block:: python

            from cilpy.problem.unconstrained import Sphere
            from cilpy.solver.ga import GA
            from cilpy.runner import ExperimentRunner

            # 1. Define the problems to test on
            problems = [Sphere(dimension=10)]

            # 2. Define the solver configurations to test
            solver_configs = [
                {
                    "class": GA,
                    "params": {
                        "name": "GA_HighMutation",
                        "population_size": 50,
                        "mutation_rate": 0.2,
                        "crossover_rate": 0.8,
                    }
                },
                {
                    "class": GA,
                    "params": {
                        "name": "GA_LowMutation",
                        "population_size": 50,
                        "mutation_rate": 0.05,
                        "crossover_rate": 0.8,
                    }
                }
            ]

            # 3. Initialize and run the experiment
            runner = ExperimentRunner(
                problems=problems,
                solver_configurations=solver_configs,
                num_runs=30,
                max_iterations=1000
            )
            runner.run_experiments()
    """

    def __init__(self,
                 problems: Sequence[Problem],
                 solver_configurations: List[Dict[str, Any]],
                 num_runs: int,
                 max_iterations: int):
        """
        Initializes the ExperimentRunner.

        Args:
            problems: A sequence of problem instances to be solved.
                Each object must implement the `Problem` interface.
            solver_configurations: A list of solver configurations.
                Each configuration is a dictionary specifying the solver class
                and its parameters. The `problem` parameter is injected
                automatically by the runner and should not be included.
            num_runs: The number of independent runs for each
                solver-problem pair.
            max_iterations: The number of iterations (`solver.step()` calls)
                per run.

        Solver Configuration Format:
            .. code-block:: python

                [
                    {
                        "class": YourSolverClass,
                        "params": {
                            "name": "UniqueSolverName",
                            "param1": value1,
                            # ... other solver hyperparameters
                        }
                    },
                    # ... more configurations
                ]
        """
        self.problems = problems
        self.solver_configurations = solver_configurations
        self.num_runs = num_runs
        self.max_iterations = max_iterations

    def run_experiments(self):
        """Executes the full suite of defined experiments.

        This method iterates through each problem and applies every configured
        solver. For each problem-solver pair, it performs `num_runs` independent
        runs, each lasting for `max_iterations`.

        Results are logged to separate CSV files, with each file named using the
        pattern: `{problem.name}_{solver_name}.out.csv`.
        """

        total_start_time = time.time()
        print("======== Starting All Experiments ========")

        # Ensure output directory exists
        os.makedirs("out", exist_ok=True)

        for problem in self.problems:
            print(f"\n--- Processing Problem: {problem.name} ---")
            for config in self.solver_configurations:
                solver_class = config["class"]
                solver_params = config["params"].copy()
                constraint_handler_config = config.get("constraint_handler")

                # Add the current problem to the solver's parameters
                current_solver_params = solver_params
                current_solver_params["problem"] = problem

                solver_name = current_solver_params.get("name")
                output_file_path = os.path.join("out", f"{problem.name}_{solver_name}.out.csv")

                print(f"\n  -> Starting Experiment: {solver_name} on {problem.name}")
                print(f"     Configuration: {self.num_runs} runs, {self.max_iterations} iterations/run.")
                print(f"     Results will be saved to: {output_file_path}")

                self._run_single_experiment(
                    solver_class,
                    current_solver_params,
                    output_file_path,
                    constraint_handler_config
                )

        total_end_time = time.time()
        print("\n======== All Experiments Finished ========")
        print(f"Total execution time: {total_end_time - total_start_time:.2f}s")

    def _is_solution_feasible(self, evaluation: Evaluation, tolerance=1e-6) -> bool:
        """
        Checks if an evaluation corresponds to a feasible solution.

        A solution is feasible if all inequality constraints are <= 0 and all
        equality constraints are approximately == 0.

        Args:
            evaluation (Evaluation): The evaluation object to check.
            tolerance (float): The tolerance for checking equality constraints.

        Returns:
            bool: True if the solution is feasible, False otherwise.
        """
        if evaluation is None:
            return False

        # Check inequality constraints: g(x) <= 0
        if evaluation.constraints_inequality:
            if any(v > 0 for v in evaluation.constraints_inequality):
                return False

        # Check equality constraints: h(x) == 0
        if evaluation.constraints_equality:
            if any(abs(v) > tolerance for v in evaluation.constraints_equality):
                return False

        return True

    def _run_single_experiment(
            self,
            solver_class: Type[Solver],
            solver_params: Dict,
            output_file: str,
            constraint_handler_config: Optional[Dict] = None):
        """Runs and logs a single experiment for a given solver on a problem.

        This internal method is called by `run_experiments`. It handles the
        instantiation of the solver for each of the `num_runs` and manages the
        iteration loop and CSV writing for a single problem-solver pair.

        The output CSV file contains the following columns:
        - `run`: The ID of the independent run (from 1 to `num_runs`).
        - `iteration`: The current iteration number (from 1 to `max_iterations`).
        - `best_fitness`: The fitness of the best solution found by the solver.
        - `is_feasible`: 1 if the best solution is feasible, 0 otherwise.
        - `feasibility_percentage`: The percentage of solutions that are
           feasible
        - `optimum_value`: The theoretical best fitness of the problem.
        - `worst_value`: The theoretical worst fitness of the problem.

        Args:
            solver_class: The solver class to be instantiated.
            solver_params: The parameters for initializing the solver (including
                the `problem` instance).
            output_file: The path to the output CSV file.
        """
        header = ["run", "iteration", "best_fitness", "is_feasible", 
        "feasibility_percentage", "optimum_value", "worst_value"]
        experiment_start_time = time.time()

        with open(output_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for run_id in range(1, self.num_runs + 1):
                run_start_time = time.time()
                print(f"     --- Starting Run {run_id}/{self.num_runs} ---")

                constraint_handler = None
                if constraint_handler_config:
                    handler_class = constraint_handler_config["class"]
                    handler_params = constraint_handler_config.get("params", {})
                    constraint_handler = handler_class(**handler_params)
                
                # Add the handler to the solver's parameters
                current_solver_params = solver_params.copy()
                current_solver_params["constraint_handler"] = constraint_handler

                # Re-instantiate the solver for each run to ensure independence
                solver = solver_class(**current_solver_params)

                for iteration in range(1, self.max_iterations + 1):
                    solver.step()
                    result = solver.get_result()

                    if result:
                        best_eval = result[0][1]
                        best_fitness = best_eval.fitness
                        is_feasible = 1 if self._is_solution_feasible(best_eval) else 0
                    else:
                        best_fitness = float('nan')
                        is_feasible = 0

                    # Safely get population evaluations
                    try:
                        all_evaluations = solver.get_population_evaluations()
                        if all_evaluations:
                            num_feasible = sum(1 for e in all_evaluations if self._is_solution_feasible(e))
                            feasibility_percentage = (num_feasible / len(all_evaluations)) * 100
                        else:
                            feasibility_percentage = 0.0
                    except NotImplementedError:
                        # If the problem doesn't implement it, log empty strings
                        feasibility_percentage = ''

                    # Safely get optimum and worst values
                    try:
                        optimum_value = solver.problem.get_optimum_value()
                        worst_value = solver.problem.get_worst_value()
                    except NotImplementedError:
                        # If the problem doesn't implement them, log empty strings
                        optimum_value = ''
                        worst_value = ''

                    # Log the data
                    writer.writerow([
                        run_id,
                        iteration,
                        best_fitness,
                        is_feasible,
                        feasibility_percentage,
                        optimum_value,
                        worst_value
                    ])

                run_end_time = time.time()
                final_result = solver.get_result()
                print(
                    f"     Run {run_id} finished in {run_end_time - run_start_time:.2f}s. "
                    f"Best fitness: {final_result[0][1].fitness if final_result else 'N/A'}"
                )

        experiment_end_time = time.time()
        solver_name = solver_params.get('name', solver_class.__name__)
        problem_name = solver_params['problem'].name
        print(f"  -> Experiment for {solver_name} on {problem_name} "
              f"finished in {experiment_end_time - experiment_start_time:.2f}s.")


if __name__ == '__main__':
    from cilpy.problem.unconstrained import Sphere, Ackley
    from cilpy.solver.pso import PSO
    from cilpy.solver.ga import GA

    # --- 1. Define the Problems ---
    dim = 3
    dom = (-5.12, 5.12)
    # A simple unconstrained problem to test the new logger
    class SimpleSphere(Problem[list[float], float]):
        def __init__(self, dimension: int):
            super().__init__(
                dimension=dimension,
                bounds=([-5.12] * dimension, [5.12] * dimension),
                name="Sphere"
            )
        def evaluate(self, solution: list[float]) -> Evaluation[float]:
            fitness = sum(x**2 for x in solution)
            return Evaluation(fitness=fitness)
        def get_optimum_value(self) -> float:
            return 0.0
        def get_worst_value(self) -> float:
            return self.dimension * (5.12 ** 2)
        def is_dynamic(self) -> Tuple[bool, bool]:
            return (False, False)

    problems_to_run = [
        SimpleSphere(dimension=dim)
    ]

    # --- 2. Define the Solver Configurations ---
    solver_configs = [
        {
            "class": GA,
            "params": {
                "name": "GA_HighMutation",
                "population_size": 30,
                "crossover_rate": 0.2,
                "mutation_rate": 0.3,
                "tournament_size": 7,
            }
        },
        {
            "class": PSO,
            "params": {
                "name": "PSO_Standard",
                "swarm_size": 30,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            }
        }
    ]

    # --- 3. Define the Experiment parameters ---
    number_of_runs = 5
    max_iter = 1000

    # --- 4. Create and run the experiments ---
    runner = ExperimentRunner(
        problems=problems_to_run,
        solver_configurations=solver_configs,
        num_runs=number_of_runs,
        max_iterations=max_iter
    )
    runner.run_experiments()
