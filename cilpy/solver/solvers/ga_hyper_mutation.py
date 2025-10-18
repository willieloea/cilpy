# cilpy/solver/solvers/ga_hyper_mutation.py

import copy
from typing import List, Tuple

from ...problem import Problem, Evaluation
from .ga import GA


class HyperMGA(GA):
    """
    A Hyper-mutation Genetic Algorithm (HyperM GA) for dynamic optimization.

    This algorithm extends the canonical GA to adapt to changing environments.
    It detects a change in the problem landscape by monitoring the fitness of
    the best solution. If the fitness degrades, it triggers a "hyper-mutation"
    phase with a significantly higher mutation rate for a fixed period to
    re-introduce diversity.

    This implementation is based on the description in Section 3.2.2 of
    Pamparà's PhD thesis.
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 hyper_mutation_rate: float,
                 hyper_period: int,
                 tournament_size: int = 2,
                 **kwargs):
        """
        Initializes the Hyper-mutation Genetic Algorithm solver.

        Args:
            problem (Problem[List[float], float]): The dynamic optimization
                problem to solve.
            name (str): the name of the solver
            population_size (int): The number of individuals in the population.
            crossover_rate (float): The base probability of crossover.
            mutation_rate (float): The standard mutation rate (pm).
            hyper_mutation_rate (float): The higher mutation rate (p_hyper)
                used when the environment changes.
            hyper_period (int): The number of generations to remain in the
                hyper-mutation state after a change is detected.
            tournament_size (int, optional): The number of individuals for
                tournament selection. Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(problem,
                         name,
                         population_size,
                         crossover_rate,
                         mutation_rate,
                         tournament_size,
                         **kwargs)

        self.hyper_mutation_rate = hyper_mutation_rate
        self.hyper_period = hyper_period

        # State tracking variables
        self.f_best: float = float('inf')
        self.hyper_count = 0
        self.is_hyper_mutation = False
        self._update_best_fitness()

    def _update_best_fitness(self):
        """Updates the tracked best fitness value from the current population."""
        current_best_eval = min(self.evaluations, key=lambda e: e.fitness)
        self.f_best = current_best_eval.fitness

    def step(self) -> None:
        """Performs one generation of the HyperM GA."""
        # --- Change Detection ---
        # Evaluate the current population to get f_test
        self.evaluations = [self.problem.evaluate(ind) for ind in self.population]
        f_test = min(e.fitness for e in self.evaluations)

        # If f_best is undefined or fitness has degraded, trigger hyper-mutation
        if self.f_best == float('inf') or f_test > self.f_best:
            self.is_hyper_mutation = True
            self.hyper_count = 0

        # --- State Management ---
        if self.is_hyper_mutation:
            current_mutation_rate = self.hyper_mutation_rate
            self.hyper_count += 1
            if self.hyper_count > self.hyper_period:
                self.is_hyper_mutation = False
        else:
            current_mutation_rate = self.mutation_rate

        # --- Standard GA Operators ---
        # 1. Selection
        parents = self._selection()

        # 2. Reproduction
        offspring = self._reproduction(parents)

        # 3. Mutation (using the current mutation rate)
        # We temporarily set self.mutation_rate for the _mutation method to use
        original_rate = self.mutation_rate
        self.mutation_rate = current_mutation_rate
        mutated_offspring = self._mutation(offspring)
        self.mutation_rate = original_rate  # Restore original rate

        # 4. Evaluate and Combine (with elitism)
        offspring_evaluations = [self.problem.evaluate(ind) for ind in mutated_offspring]

        best_current_idx = min(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        best_individual = self.population[best_current_idx]
        best_evaluation = self.evaluations[best_current_idx]

        self.population = mutated_offspring
        self.evaluations = offspring_evaluations

        worst_new_idx = max(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        self.population[worst_new_idx] = best_individual
        self.evaluations[worst_new_idx] = best_evaluation

        # Update f_best for the next iteration
        self._update_best_fitness()
