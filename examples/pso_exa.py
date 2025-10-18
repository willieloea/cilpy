# examples/pso_exa.py

import time
import csv

# --- Import cilpy components ---
from cilpy.problem.functions import Sphere, Ackley
from cilpy.solver.solvers.pso import PSO
from cilpy.solver.solvers.ga import GA

"""
An example of using the ExperimentRunner to run PSO on the Sphere function.
"""
# 1. Define the Problem
dim = 3
dom = (-5.12, 5.12)
pro1 = Sphere(dimension=dim, domain=dom)
pro2 = Ackley(dimension=dim, domain=dom)

problems = [pro1, pro2]

for problem in problems:
    # 2. Define the Solver parameters
    pso_class = PSO
    pso_params = {
        "problem": problem,
        "name": "PSO",
        "swarm_size": 30,
        "w": 0.7298,
        "c1": 1.49618,
        "c2": 1.49618,
        "k": 1, # The neighborhood size (1 neighbor on each side)
    }

    ga_class = GA
    ga_params = {
        "problem": problem,
        "name": "GA",
        "population_size": 30,
        "crossover_rate": 0.2,
        "mutation_rate": 0.2,
        "tournament_size": 7,
    }

    solvers = {
        pso_class: pso_params,
        ga_class: ga_params
    }

    # 3. Define the Experiment parameters
    num_runs=5
    max_iterations=1000

    for solver_class, solver_params in solvers.items():
        output_file_path=f"{problem.name}_{solver_params.get("name")}.out.csv"

        # 4. Create and run the experiment
        print(f"Starting experiment: {solver_params.get("name")} on {problem.name}")
        print(f"Configuration: {num_runs} runs, {max_iterations} iterations/run.")
        print(f"Results will be saved to: {output_file_path}")

        header = ["run", "iteration", "result"]

        with open(output_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            total_start_time = time.time()
            for run_id in range(1, num_runs + 1):
                run_start_time = time.time()
                print(f"--- Starting Run {run_id}/{num_runs} ---")

                # Re-instantiate the solver for each run to ensure independence
                solver = solver_class(**solver_params)

                for iteration in range(1, max_iterations + 1):
                    solver.step()
                    result = solver.get_result()

                    # Log data for this iteration
                    writer.writerow([run_id, iteration, result])

                run_end_time = time.time()
                result = solver.get_result()
                print(
                    f"Run {run_id} finished in {run_end_time - run_start_time:.2f}s. "
                    f"Best result: {result}"
                )

        total_end_time = time.time()
        print("\n--- Experiment Finished ---")
        print(f"Total execution time: {total_end_time - total_start_time:.2f}s\n\n")
