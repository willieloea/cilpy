import time
import csv
from typing import List, Dict, Type

from cilpy.problem import Problem
from cilpy.solver import Solver


class ExperimentRunner:
    """
    A generic runner for conducting computational intelligence experiments.

    This class orchestrates the process of running multiple optimization algorithms
    (solvers) on a collection of problems for a specified number of independent runs
    and iterations. It handles the setup, execution, and result logging for
    each experiment.
    """

    def __init__(self,
                 problems: List[Problem],
                 solvers: Dict[Type[Solver], Dict],
                 num_runs: int,
                 max_iterations: int):
        """
        Initializes the ExperimentRunner.

        Args:
            problems (List[Problem]): A list of problem instances to be solved.
                Each problem must implement the `Problem` interface.
            solvers (Dict[Type[Solver], Dict]): A dictionary where keys are solver
                classes (e.g., `PSO`, `GA`) and values are dictionaries of their
                respective parameters. The 'problem' parameter will be set by
                the runner.
            num_runs (int): The number of independent runs to perform for each
                solver-problem pair.
            max_iterations (int): The number of iterations to run each solver for
                in a single run.
        """
        self.problems = problems
        self.solvers = solvers
        self.num_runs = num_runs
        self.max_iterations = max_iterations

    def run_experiments(self):
        """
        Executes the full suite of experiments defined during initialization.

        This method iterates through each problem and applies every configured
        solver to it, performing the specified number of runs and iterations.
        Results are logged to separate CSV files for each problem-solver pair.
        """
        total_start_time = time.time()
        print("======== Starting All Experiments ========")

        for problem in self.problems:
            print(f"\n--- Processing Problem: {problem.name} ---")
            for solver_class, solver_params in self.solvers.items():
                # Update solver parameters with the current problem
                current_solver_params = solver_params.copy()
                current_solver_params["problem"] = problem

                solver_name = current_solver_params.get("name", solver_class.__name__)
                output_file_path = f"{problem.name}_{solver_name}.out.csv"

                print(f"\n  -> Starting Experiment: {solver_name} on {problem.name}")
                print(f"     Configuration: {self.num_runs} runs, {self.max_iterations} iterations/run.")
                print(f"     Results will be saved to: {output_file_path}")

                self._run_single_experiment(solver_class, current_solver_params, output_file_path)

        total_end_time = time.time()
        print("\n======== All Experiments Finished ========")
        print(f"Total execution time: {total_end_time - total_start_time:.2f}s")

    def _run_single_experiment(self, solver_class: Type[Solver], solver_params: Dict, output_file: str):
        """
        Runs and logs a single experiment for a given solver and problem.

        Args:
            solver_class (Type[Solver]): The class of the solver to be used.
            solver_params (Dict): The parameters for initializing the solver.
            output_file (str): The path to the CSV file where results will be saved.
        """
        header = ["run", "iteration", "result"]
        experiment_start_time = time.time()

        with open(output_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for run_id in range(1, self.num_runs + 1):
                run_start_time = time.time()
                print(f"     --- Starting Run {run_id}/{self.num_runs} ---")

                # Re-instantiate the solver for each run to ensure independence
                solver = solver_class(**solver_params)

                for iteration in range(1, self.max_iterations + 1):
                    solver.step()
                    result = solver.get_result()

                    # Log data for this iteration
                    writer.writerow([run_id, iteration, result])

                run_end_time = time.time()
                final_result = solver.get_result()
                print(
                    f"     Run {run_id} finished in {run_end_time - run_start_time:.2f}s. "
                    f"Best result: {final_result}"
                )

        experiment_end_time = time.time()
        print(f"  -> Experiment for {solver_params.get('name')} on {solver_params['problem'].name} "
              f"finished in {experiment_end_time - experiment_start_time:.2f}s.")


if __name__ == '__main__':
    from cilpy.problem.functions import Sphere, Ackley
    from cilpy.solver.solvers.pso import PSO
    from cilpy.solver.solvers.ga import GA

    # --- 1. Define the Problems ---
    dim = 3
    dom = (-5.12, 5.12)
    problems_to_run = [
        Sphere(dimension=dim, domain=dom),
        Ackley(dimension=dim, domain=dom)
    ]

    # --- 2. Define the Solvers and their parameters ---
    # Note: The 'problem' parameter is omitted here as the runner will assign it.
    solvers_to_run = {
        PSO: {
            "name": "PSO",
            "swarm_size": 30,
            "w": 0.7298,
            "c1": 1.49618,
            "c2": 1.49618,
            "k": 1,  # The neighborhood size (1 neighbor on each side)
        },
        GA: {
            "name": "GA",
            "population_size": 30,
            "crossover_rate": 0.2,
            "mutation_rate": 0.2,
            "tournament_size": 7,
        }
    }

    # --- 3. Define the Experiment parameters ---
    number_of_runs = 5
    max_iter = 1000

    # --- 4. Create and run the experiments ---
    runner = ExperimentRunner(
        problems=problems_to_run,
        solvers=solvers_to_run,
        num_runs=number_of_runs,
        max_iterations=max_iter
    )
    runner.run_experiments()
