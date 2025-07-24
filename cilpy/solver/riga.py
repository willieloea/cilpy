# cilpy/solver/riga.py

import random
from typing import List, Tuple, Any

from ..problem import Problem
from . import Solver

# =============================================================================
# Helper Functions for the Genetic Algorithm
# =============================================================================


def _tournament_selection(
    population: List[List[float]], fitnesses: List[float], tournament_size: int
) -> List[float]:
    """
    Selects an individual from the population using tournament selection.
    """
    if not population:
        raise ValueError("Population cannot be empty for tournament selection.")

    best_participant_idx = -1
    best_fitness = float("inf")

    # Select random participants for the tournament
    participants_indices = random.sample(range(len(population)), tournament_size)

    for idx in participants_indices:
        if fitnesses[idx] < best_fitness:
            best_fitness = fitnesses[idx]
            best_participant_idx = idx

    return population[best_participant_idx]


def _blend_crossover(
    parent1: List[float], parent2: List[float], alpha: float = 0.5
) -> Tuple[List[float], List[float]]:
    """
    Performs blend crossover (BLX-alpha) on two parents.
    """
    dim = len(parent1)
    child1, child2 = [0.0] * dim, [0.0] * dim

    for i in range(dim):
        d = abs(parent1[i] - parent2[i])
        lower = min(parent1[i], parent2[i]) - alpha * d
        upper = max(parent1[i], parent2[i]) + alpha * d

        child1[i] = random.uniform(lower, upper)
        child2[i] = random.uniform(lower, upper)

    return child1, child2


def _uniform_mutation(
    individual: List[float], bounds: Tuple[List[float], List[float]], p_mutation: float
) -> List[float]:
    """
    Performs uniform mutation on an individual.
    Each gene has a `p_mutation` chance to be replaced by a new random value
    within its bounds.
    """
    mutated_individual = individual[:]
    lower_bounds, upper_bounds = bounds

    for i in range(len(mutated_individual)):
        if random.random() < p_mutation:
            mutated_individual[i] = random.uniform(lower_bounds[i], upper_bounds[i])

    return mutated_individual


# =============================================================================
# Main RIGA Solver Class
# =============================================================================


class RIGASolver(Solver[List[float]]):
    """
    Random Immigrants Genetic Algorithm (RIGA) solver.

    This solver implements a standard Genetic Algorithm (GA) with an additional
    mechanism to maintain diversity, as described by Grefenstette. At the end
    of each generation, a fixed proportion of the population's worst-performing
    individuals are replaced by new, randomly generated individuals (immigrants).
    This helps prevent premature convergence and allows the algorithm to adapt
    in dynamic environments.

    This implementation is adapted for dynamic optimization problems by
    re-evaluating the entire population's fitness if the environment changes.
    """

    def __init__(
        self,
        problem: Problem[List[float]],
        population_size: int = 50,
        p_crossover: float = 0.9,
        p_mutation: float = 0.05,
        p_immigrants: float = 0.1,
        tournament_size: int = 3,
        **kwargs: Any
    ):
        """
        Initializes the RIGA solver.

        Args:
            problem: The optimization problem to solve.
            population_size: Number of individuals in the population.
            p_crossover: The probability of performing crossover.
            p_mutation: The probability of mutating each gene in an offspring.
            p_immigrants: The proportion of the population to be replaced by
                          random immigrants each generation (Pim).
            tournament_size: The number of individuals participating in each
                             selection tournament.
            **kwargs: Additional parameters (ignored).
        """
        super().__init__(problem, **kwargs)
        if population_size % 2 != 0:
            # Ensures population size is even for straightforward crossover
            population_size += 1

        self.population_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_immigrants = p_immigrants
        self.tournament_size = tournament_size
        self.num_immigrants = int(self.population_size * self.p_immigrants)

        # Problem-specific attributes
        self.objective = self.problem.get_objective_functions()[0]
        self.bounds = self.problem.get_bounds()
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

        # Initialize population and fitness
        self.population = [
            self.problem.initialize_solution() for _ in range(self.population_size)
        ]
        self.fitnesses = [self.objective(ind) for ind in self.population]

        # Find and store the initial best solution
        best_idx = min(range(self.population_size), key=lambda i: self.fitnesses[i])
        self.gbest_position = self.population[best_idx][:]
        self.gbest_value = self.fitnesses[best_idx]

    def _clamp_position(self, position: List[float]) -> List[float]:
        """Clamps a position to the problem's bounds."""
        lower, upper = self.bounds
        return [max(l, min(x, u)) for x, l, u in zip(position, lower, upper)]

    def step(self) -> None:
        """Performs one generation of the RIGA algorithm."""

        # Re-evaluate fitness if the environment is dynamic
        if self.is_dynamic or self.is_constrained_dynamic:
            self.fitnesses = [self.objective(ind) for ind in self.population]
            self.gbest_value = self.objective(self.gbest_position)

            # Re-check for a new global best in the current population
            current_best_idx = min(
                range(self.population_size), key=lambda i: self.fitnesses[i]
            )
            if self.fitnesses[current_best_idx] < self.gbest_value:
                self.gbest_position = self.population[current_best_idx][:]
                self.gbest_value = self.fitnesses[current_best_idx]

        # Generate offspring via selection, crossover, and mutation
        offspring_population = []
        for _ in range(self.population_size // 2):
            # Selection
            parent1 = _tournament_selection(
                self.population, self.fitnesses, self.tournament_size
            )
            parent2 = _tournament_selection(
                self.population, self.fitnesses, self.tournament_size
            )

            # Crossover
            if random.random() < self.p_crossover:
                child1, child2 = _blend_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutation and clamping
            child1 = _uniform_mutation(child1, self.bounds, self.p_mutation)
            child2 = _uniform_mutation(child2, self.bounds, self.p_mutation)
            offspring_population.append(self._clamp_position(child1))
            offspring_population.append(self._clamp_position(child2))

        # Evaluate offspring
        offspring_fitnesses = [self.objective(ind) for ind in offspring_population]

        # Environmental Selection (Elitism)
        # Combine parent and offspring populations and select the best
        combined_population = self.population + offspring_population
        combined_fitnesses = self.fitnesses + offspring_fitnesses

        sorted_indices = sorted(
            range(len(combined_fitnesses)), key=lambda k: combined_fitnesses[k]
        )

        next_population = [
            combined_population[i] for i in sorted_indices[: self.population_size]
        ]
        next_fitnesses = [
            combined_fitnesses[i] for i in sorted_indices[: self.population_size]
        ]

        self.population = next_population
        self.fitnesses = next_fitnesses

        # Introduce Random Immigrants
        if self.num_immigrants > 0:
            # Find the indices of the worst individuals to be replaced
            # Population is already sorted best-to-worst from the previous step
            worst_indices = range(
                self.population_size - self.num_immigrants, self.population_size
            )

            for i in worst_indices:
                # Generate a new random immigrant
                immigrant = self.problem.initialize_solution()
                immigrant_fitness = self.objective(immigrant)

                # Replace the worst individual
                self.population[i] = immigrant
                self.fitnesses[i] = immigrant_fitness

        # Update the global best solution
        # The best individual is always at index 0 after sorting
        if self.fitnesses[0] < self.gbest_value:
            self.gbest_value = self.fitnesses[0]
            self.gbest_position = self.population[0][:]

    def get_best(self) -> Tuple[List[float], List[float]]:
        """Returns the best solution and its objective value found so far."""
        return self.gbest_position, [self.gbest_value]
