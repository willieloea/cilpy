# cilpy/solver/solvers/ga.py

import random
import copy
from typing import List, Tuple

from ...problem import Problem, Evaluation, SolutionType, FitnessType
from .. import Solver


class GA(Solver[List[float], float]):
    """
    A canonical Genetic Algorithm (GA) for single-objective optimization.

    This implementation is based on the structure outlined in Section 3.1.1
    of the reference document. It follows a generational model with selection,
    reproduction (crossover), and mutation operators.

    The algorithm uses:
    - Tournament selection to choose parents.
    - Single-point crossover for reproduction.
    - Gaussian mutation to introduce genetic diversity.
    - Elitism to preserve the best solution across generations.
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 tournament_size: int = 2,
                 **kwargs):
        """
        Initializes the Genetic Algorithm solver.

        Args:
            problem (Problem[List[float], float]): The optimization problem to solve.
            name (str): the name of the solver
            population_size (int): The number of individuals in the population.
            crossover_rate (float): The probability of crossover (pc) occurring
                between two parents.
            mutation_rate (float): The probability of mutation (pm) for each
                gene in an offspring.
            tournament_size (int, optional): The number of individuals to select
                for each tournament. Defaults to 2.
            **kwargs: Additional keyword arguments (not used in this canonical GA).
        """
        super().__init__(problem, name)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

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

    def _selection(self) -> List[List[float]]:
        """Performs tournament selection to choose parents."""
        parents = []
        for _ in range(self.population_size):
            tournament = random.sample(list(range(self.population_size)), self.tournament_size)
            winner_idx = min(tournament, key=lambda i: self.evaluations[i].fitness)
            parents.append(self.population[winner_idx])
        return parents

    def _reproduction(self, parents: List[List[float]]) -> List[List[float]]:
        """Creates offspring through single-point crossover."""
        offspring = []
        for i in range(0, self.population_size, 2):
            p1 = parents[i]
            # Ensure there's a second parent for crossover
            p2 = parents[i + 1] if i + 1 < self.population_size else parents[0]

            if random.random() < self.crossover_rate:
                crossover_point = random.randint(1, self.problem.dimension - 1)
                c1 = p1[:crossover_point] + p2[crossover_point:]
                c2 = p2[:crossover_point] + p1[crossover_point:]
                offspring.extend([c1, c2])
            else:
                offspring.extend([copy.deepcopy(p1), copy.deepcopy(p2)])
        return offspring[:self.population_size]

    def _mutation(self, offspring: List[List[float]]) -> List[List[float]]:
        """Applies Gaussian mutation to offspring."""
        lower_bounds, upper_bounds = self.problem.bounds
        for individual in offspring:
            for i in range(self.problem.dimension):
                if random.random() < self.mutation_rate:
                    # Add noise from a Gaussian distribution with mean 0
                    mutation_value = random.gauss(0, (upper_bounds[i] - lower_bounds[i]) * 0.1)
                    individual[i] += mutation_value
                    # Clamp the value to within the problem bounds
                    individual[i] = max(lower_bounds[i], min(individual[i], upper_bounds[i]))
        return offspring

    def step(self) -> None:
        """Performs one generation of the Genetic Algorithm."""
        # 1. Selection
        parents = self._selection()

        # 2. Reproduction
        offspring = self._reproduction(parents)

        # 3. Mutation
        mutated_offspring = self._mutation(offspring)

        # 4. Evaluate new offspring
        offspring_evaluations = [self.problem.evaluate(ind) for ind in mutated_offspring]

        # 5. Combine (create next generation) with elitism
        # Find the best individual from the current generation
        best_current_idx = min(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        best_individual = self.population[best_current_idx]
        best_evaluation = self.evaluations[best_current_idx]

        # The new population is the mutated offspring
        self.population = mutated_offspring
        self.evaluations = offspring_evaluations

        # Find the worst individual in the new generation and replace it with the best from the previous
        worst_new_idx = max(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        self.population[worst_new_idx] = best_individual
        self.evaluations[worst_new_idx] = best_evaluation

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """Returns the best solution found in the current population."""
        best_idx = min(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        best_solution = self.population[best_idx]
        best_evaluation = self.evaluations[best_idx]
        return [(best_solution, best_evaluation)]
