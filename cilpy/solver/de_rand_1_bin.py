# cilpy/solver/de_rand_1_bin.py

import random
from typing import List, Tuple, Any

from ..problem import Problem
from . import Solver


class DifferentialEvolutionSolver(Solver[List[float]]):
    """
    Implements the classic DE/rand/1/bin Differential Evolution algorithm.

    - **rand**: The base vector for mutation is chosen randomly.
    - **1**: One difference vector is used for mutation.
    - **bin**: Binomial (uniform) crossover is used.

    This solver is adapted for dynamic optimization problems by re-evaluating the
    entire population's fitness at the start of each step if the environment
    has changed.
    """

    def __init__(
        self,
        problem: Problem[List[float]],
        population_size: int = 50,
        scale_factor: float = 0.8,
        crossover_prob: float = 0.9,
        **kwargs: Any
    ):
        """
        Initializes the Differential Evolution solver.

        Args:
            problem: The optimization problem to solve.
            population_size: The number of individuals in the population (NP).
            scale_factor: The mutation factor (F), usually in [0.4, 1.0].
            crossover_prob: The crossover probability (CR), usually in [0, 1].
            **kwargs: Additional parameters (ignored).
        """
        super().__init__(problem, **kwargs)
        self.population_size = population_size
        self.f = scale_factor  # F
        self.cr = crossover_prob  # CR
        self.objective = self.problem.get_objective_functions()[0]
        self.dimension = self.problem.get_dimension()

        # Initialize population and evaluate fitness
        self.population = [
            self.problem.initialize_solution() for _ in range(self.population_size)
        ]
        self.fitness = [self.objective(ind) for ind in self.population]

        # Find initial global best
        best_idx = min(range(self.population_size), key=lambda i: self.fitness[i])
        self.gbest_position = self.population[best_idx]
        self.gbest_value = self.fitness[best_idx]

        # Store dynamic status
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

    def _clamp_position(self, position: List[float]) -> List[float]:
        """Clamps a position to the problem's bounds."""
        lower, upper = self.problem.get_bounds()
        return [max(l, min(x, u)) for x, l, u in zip(position, lower, upper)]

    def step(self) -> None:
        """Performs one generation of the DE algorithm."""

        # Re-evaluate population if the environment is dynamic
        if self.is_dynamic or self.is_constrained_dynamic:
            self.fitness = [self.objective(ind) for ind in self.population]
            # After re-evaluation, find the new best solution
            best_idx = min(range(self.population_size), key=lambda i: self.fitness[i])
            self.gbest_position = self.population[best_idx]
            self.gbest_value = self.fitness[best_idx]

        # Main DE loop for one generation
        for i in range(self.population_size):
            # --- Mutation ---
            # Select three distinct individuals other than the current one
            indices = list(range(self.population_size))
            indices.remove(i)
            r1, r2, r3 = random.sample(indices, 3)

            # Create the mutant vector v = x_r1 + F * (x_r2 - x_r3)
            mutant_vector = [
                self.population[r1][d]
                + self.f * (self.population[r2][d] - self.population[r3][d])
                for d in range(self.dimension)
            ]
            mutant_vector = self._clamp_position(mutant_vector)

            # --- Crossover ---
            # Create the trial vector by binomial crossover
            trial_vector = []
            j_rand = random.randrange(self.dimension)
            for d in range(self.dimension):
                if random.random() < self.cr or d == j_rand:
                    trial_vector.append(mutant_vector[d])
                else:
                    trial_vector.append(self.population[i][d])

            # --- Selection ---
            trial_fitness = self.objective(trial_vector)

            # If the trial vector is better, it replaces the target vector
            if trial_fitness < self.fitness[i]:
                self.population[i] = trial_vector
                self.fitness[i] = trial_fitness

                # Update the global best if necessary
                if trial_fitness < self.gbest_value:
                    self.gbest_value = trial_fitness
                    self.gbest_position = trial_vector

    def get_best(self) -> Tuple[List[float], List[float]]:
        """Returns the best solution and its objective value found so far."""
        return self.gbest_position, [self.gbest_value]
