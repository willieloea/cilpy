# cilpy/solver/hmga.py

import random
from typing import List, Tuple, Any

from ...cilpy.problem import Problem
from ...cilpy.solver import Solver


class HyperMutationGA(Solver[List[float]]):
    """
    Hyper-mutation Genetic Algorithm (HyperM) Solver.

    This algorithm is an extension of the canonical Genetic Algorithm (GA)
    designed to introduce diversity in dynamically changing optimization
    landscapes. It was proposed by Cobb [43].

    The core mechanism of HyperM is to switch from a standard, low mutation
    rate to a very high "hyper-mutation" rate when a degradation in the
    objective function value is detected. This degradation is identified by
    tracking the best-so-far solution's fitness. The algorithm remains in
    hyper-mutation mode for a fixed number of generations before returning to
    the standard rate, allowing the population to explore new regions of the
    search space.

    This implementation uses tournament selection, uniform crossover, and
    gaussian mutation as its core GA operators.
    """

    def __init__(
        self,
        problem: Problem[List[float]],
        population_size: int = 50,
        crossover_prob: float = 0.9,
        pm: float = 0.01,
        phyper: float = 0.2,
        hyper_total: int = 5,
        tournament_size: int = 3,
        **kwargs: Any
    ):
        """
        Initializes the Hyper-mutation Genetic Algorithm solver.

        Args:
            problem: The optimization problem to solve.
            population_size: The number of individuals in the population.
            crossover_prob: The probability of performing crossover.
            pm: The standard mutation rate.
            phyper: The hyper-mutation rate (must be > pm).
            hyper_total: The number of generations to stay in hyper-mutation
                         mode after it is triggered.
            tournament_size: The number of individuals to select for tournament
                             selection.
            **kwargs: Additional parameters (ignored).
        """
        super().__init__(problem, **kwargs)
        if not (0 <= pm < phyper <= 1.0):
            raise ValueError("Mutation rates must satisfy 0 <= pm < phyper <= 1.0")

        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.pm = pm
        self.phyper = phyper
        self.hyper_total = hyper_total
        self.tournament_size = tournament_size

        self.objective = self.problem.get_objective_functions()[0]
        self.dimension = self.problem.get_dimension()
        self.bounds = self.problem.get_bounds()

        # Algorithm state variables from Algorithm 3.4
        self.is_hyper_mutating = False
        self.hyper_count = 0
        self.fbest = float(
            "inf"
        )  # Tracks the best fitness from the last 'stable' generation

        # Initialize population and best solution tracking
        self.population = [
            self.problem.initialize_solution() for _ in range(self.population_size)
        ]
        self.fitness_values = [self.objective(ind) for ind in self.population]

        best_idx = min(
            range(self.population_size), key=lambda i: self.fitness_values[i]
        )
        self.gbest_solution = self.population[best_idx][:]
        self.gbest_value = self.fitness_values[best_idx]

        # Store dynamic status to avoid repeated checks
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

    def _clamp_position(self, position: List[float]) -> List[float]:
        """Clamps a position to the problem's bounds."""
        lower, upper = self.bounds
        return [max(l, min(x, u)) for x, l, u in zip(position, lower, upper)]

    def _tournament_selection(self) -> int:
        """Performs tournament selection and returns the index of the winner."""
        best_idx = -1
        for _ in range(self.tournament_size):
            idx = random.randint(0, self.population_size - 1)
            if (
                best_idx == -1
                or self.fitness_values[idx] < self.fitness_values[best_idx]
            ):
                best_idx = idx
        return best_idx

    def _uniform_crossover(
        self, parent1: List[float], parent2: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Performs uniform crossover on two parents."""
        child1, child2 = parent1[:], parent2[:]
        for i in range(self.dimension):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def _mutate(self, individual: List[float], mutation_rate: float) -> List[float]:
        """Mutates an individual using Gaussian perturbation."""
        mutated_individual = individual[:]
        for i in range(self.dimension):
            if random.random() < mutation_rate:
                # Add a small random value from a Gaussian distribution
                mutated_individual[i] += random.gauss(
                    0, 0.1 * (self.bounds[1][i] - self.bounds[0][i])
                )
        return self._clamp_position(mutated_individual)

    def step(self) -> None:
        """Performs one generation of the HyperM-GA algorithm."""

        # Re-evaluate memory if the environment is dynamic
        if self.is_dynamic or self.is_constrained_dynamic:
            self.gbest_value = self.objective(self.gbest_solution)
            # Re-evaluate entire population as well
            self.fitness_values = [self.objective(ind) for ind in self.population]

        # Change Detection
        ftest = min(self.fitness_values)

        # Initialize fbest on the first iteration (fbest = undefined)
        if self.fbest == float("inf"):
            self.fbest = ftest

        # Trigger hyper-mutation if performance degrades
        if ftest > self.fbest:
            self.is_hyper_mutating = True
            self.hyper_count = 0

        # Stop hyper-mutation after hyper_total generations
        if self.is_hyper_mutating and self.hyper_count >= self.hyper_total:
            self.is_hyper_mutating = False
            self.hyper_count = 0

        # Determine current mutation rate
        current_mutation_rate = self.phyper if self.is_hyper_mutating else self.pm

        # Generate next generation (Selection, Crossover, Mutation)
        offspring_population = []
        for _ in range(self.population_size // 2):
            # Selection
            p1_idx = self._tournament_selection()
            p2_idx = self._tournament_selection()
            parent1, parent2 = self.population[p1_idx], self.population[p2_idx]

            # Crossover
            if random.random() < self.crossover_prob:
                child1, child2 = self._uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutation
            offspring_population.append(self._mutate(child1, current_mutation_rate))
            offspring_population.append(self._mutate(child2, current_mutation_rate))

        # Evaluate new offspring
        offspring_fitness = [self.objective(ind) for ind in offspring_population]

        # Survival Selection (Elitism: combine parents and offspring)
        combined_population = self.population + offspring_population
        combined_fitness = self.fitness_values + offspring_fitness

        # Sort combined population by fitness and select the best
        sorted_indices = sorted(
            range(len(combined_fitness)), key=lambda k: combined_fitness[k]
        )

        self.population = [
            combined_population[i] for i in sorted_indices[: self.population_size]
        ]
        self.fitness_values = [
            combined_fitness[i] for i in sorted_indices[: self.population_size]
        ]

        # Update best-so-far solution
        if self.fitness_values[0] < self.gbest_value:
            self.gbest_value = self.fitness_values[0]
            self.gbest_solution = self.population[0][:]

        # Update fbest (the tracked optimum)
        self.fbest = self.fitness_values[0]

        # Update hyper-mutation counter
        if self.is_hyper_mutating:
            self.hyper_count += 1

    def get_best(self) -> Tuple[List[float], List[float]]:
        """Returns the best solution and its objective value found so far."""
        return self.gbest_solution, [self.gbest_value]
