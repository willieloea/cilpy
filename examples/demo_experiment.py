# examples/demo_experiment.py
"""
This file aims to reproduce results obtained by Gary Pampara in his PhD thesis.

| Parameter | Static    | Progressive | Abrupt    | Chaotic   |
| Domain    | [0, 100]  | [0, 100]    | [0, 100]  | [0, 100]  |
| Dim       | 5         | 5           | 5         | 5         |
| Peak c    | 10        | 10          | 10        | 10        |
| Peak h    | [30, 70]  | [30, 70]    | [30, 70]  | [30, 70]  |
| Peak w    | [1, 12]   | [1, 12]     | [1, 12]   | [1, 12]   |
| hSeverity | 1         | 1           | 10        | 10        |
| wSeverity | 0.05      | 0.05        | 0.05      | 0.05      |
| C sev     | 1         | 1           | 10        | 10        |
| lambda    | 0         | 0           | 0         | 0         |
| C freq    | ∞         | 20          | 100       | 30        |
    ConstrainedMovingPeaksBenchmark(
        f_params=sta_params,
        g_params=sta_params,
        name="CMPB_STA_STA"
    ),
    ConstrainedMovingPeaksBenchmark(
        f_params=abr_params,
        g_params=abr_params,
        name="CMPB_A3L_A3L"
    ),
    ConstrainedMovingPeaksBenchmark(
        f_params=pro_params,
        g_params=pro_params,
        name="CMPB_PRO_PRO"
    ),
    ConstrainedMovingPeaksBenchmark(
        f_params=cha_params,
        g_params=cha_params,
        name="CMPB_CHA_CHA"
    ),
"""

from cilpy.runner import ExperimentRunner
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark, generate_mpb_configs
from cilpy.solver.ga import RIGA
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.chm.alpha_constraint import AlphaConstraintHandler

# --- 1. Define the Problems ---
all_mpb_configs = generate_mpb_configs(dimension=5)
a1r_params = all_mpb_configs['A1R']
a3r_params = all_mpb_configs['A3R']
c1r_params = all_mpb_configs['C1R']
c3r_params = all_mpb_configs['C3R']
p1r_params = all_mpb_configs['P1R']
p3r_params = all_mpb_configs['P3R']
sta_params = all_mpb_configs['STA']

problems_to_run = [
    ConstrainedMovingPeaksBenchmark(
        f_params=a1r_params,
        g_params=a1r_params,
        name="CMPB_A1R_A1R"
    ),
    ConstrainedMovingPeaksBenchmark(
        f_params=a3r_params,
        g_params=a3r_params,
        name="CMPB_A3R_A3R"
    ),
    ConstrainedMovingPeaksBenchmark(
        f_params=p3r_params,
        g_params=p3r_params,
        name="CMPB_P3R_P3R"
    ),
    ConstrainedMovingPeaksBenchmark(
        f_params=c1r_params,
        g_params=c1r_params,
        name="CMPB_C1R_C1R"
    ),
    ConstrainedMovingPeaksBenchmark(
        f_params=c3r_params,
        g_params=c3r_params,
        name="CMPB_C3R_C3R"
    ),
]

# --- 2. Define the Solvers and their parameters ---
solver_configs = [
    {
        # This is the co-evolutionary solver configuration
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "RIGA_CCLS",
            "objective_solver_class": RIGA,
            "multiplier_solver_class": RIGA,
            "objective_solver_params": {
                "name": "ObjectiveGA",
                "population_size": 50,
                "crossover_rate": 0.1,
                "mutation_rate": 0.15,
                "immigrant_rate": 0.3,
                "tournament_size": 3,
            },
            "multiplier_solver_params": {
                "name": "MultiplierGA",
                "population_size": 50,
                "crossover_rate": 0.1,
                "mutation_rate": 0.15,
                "immigrant_rate": 0.3,
                "tournament_size": 3,
            }
        }
    },
    {
        # This is the alpha-constraint solver configuration
        "class": RIGA,
        "params": {
            "name": "RIGA_AlphaConstraint",
            "population_size": 50,
            "crossover_rate": 0.1,
            "mutation_rate": 0.15,
            "immigrant_rate": 0.3,
            "tournament_size": 3,
        },
        "constraint_handler": {
            "class": AlphaConstraintHandler,
            "params": {
                "alpha": 0.95,
                "b_inequality": 5.0
            }
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 30
max_iter = 1000

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
