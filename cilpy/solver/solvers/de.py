# cilpy/solver/solvers/de.py

import random
import copy
from typing import List, Tuple

from ...problem import Problem, Evaluation, SolutionType, FitnessType
from .. import Solver


class DE(Solver[List[float], float]):
    """
    A canonical Differential Evolution (DE) solver for single-objective optimization.

    This implementation follows the `DE/rand/1/bin` scheme described in
    Section 3.1.2 of Pamparà's PhD thesis. It creates a trial vector for each
    member of the population and replaces the member if the trial vector has
    better or equal fitness.

    The algorithm uses:
    - `rand` strategy for selecting vectors for mutation.
    - `1` difference vector in the mutation step.
    - `bin` (binomial) crossover.
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 population_size: int,
                 crossover_rate: float,
                 f_weight: float,
                 **kwargs):
        """
        Initializes the Differential Evolution solver.

        Args:
            problem (Problem[List[float], float]): The optimization problem to
                solve.
            name (str): the name of the solver
            population_size (int): The number of individuals (ns) in the
                population.
            crossover_rate (float): The crossover probability (CR) in the range
                [0, 1].
            f_weight (float): The differential weight (F) for mutation,
                typically in the range [0, 2].
            **kwargs: Additional keyword arguments (not used in this canonical
                DE).
        """
        super().__init__(problem, name)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.f_weight = f_weight

        # Initialize population
        self.population = self._initialize_population()
        self.evaluations = [self.problem.evaluate(ind) for ind in self.population]

    def _initialize_population(self) -> List[List[float]]:
        """Creates the initial population with random solutions."""
        population = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(self.population_size):
            individual = [random.uniform(lower_bounds[i], upper_bounds[i])
                          for i in range(self.problem.dimension)]
            population.append(individual)
        return population

    def step(self) -> None:
        """Performs one generation of the Differential Evolution algorithm."""
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(self.population_size):
            target_vector = self.population[i]
            target_eval = self.evaluations[i]

            # 1. Mutation (Create Donor Vector) - DE/rand/1
            # Select three distinct individuals other than the target
            indices = list(range(self.population_size))
            indices.remove(i)
            r1, r2, r3 = random.sample(indices, 3)

            x_r1 = self.population[r1]
            x_r2 = self.population[r2]
            x_r3 = self.population[r3]

            donor_vector = [
                x_r1[j] + self.f_weight * (x_r2[j] - x_r3[j])
                for j in range(self.problem.dimension)
            ]

            # 2. Recombination (Create Trial Vector) - Binomial Crossover
            trial_vector = [0.0] * self.problem.dimension
            j_rand = random.randrange(self.problem.dimension)
            for j in range(self.problem.dimension):
                if random.random() < self.crossover_rate or j == j_rand:
                    trial_vector[j] = donor_vector[j]
                else:
                    trial_vector[j] = target_vector[j]

            # Ensure trial vector is within bounds
            for j in range(self.problem.dimension):
                trial_vector[j] = max(lower_bounds[j], min(trial_vector[j], upper_bounds[j]))

            # 3. Selection
            trial_eval = self.problem.evaluate(trial_vector)

            # If the trial vector is better or equal, it replaces the target vector
            if trial_eval.fitness <= target_eval.fitness:
                self.population[i] = trial_vector
                self.evaluations[i] = trial_eval

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """Returns the best solution found in the current population."""
        best_idx = min(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        best_solution = self.population[best_idx]
        best_evaluation = self.evaluations[best_idx]
        return [(best_solution, best_evaluation)]
