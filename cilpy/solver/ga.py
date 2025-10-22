# cilpy/solver/solvers/ga.py

import random
import copy
from typing import List, Tuple

from ...problem import Problem, Evaluation
from .. import Solver


class GA(Solver[List[float], float]):
    """
    A canonical Genetic Algorithm (GA) for single-objective optimization.

    This implementation is based on the structure outlined in Section 3.1.1 of
    Pamparà's PhD thesis. It follows a generational model with selection,
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

            # Crossover is only possible if the dimension is > 1.
            if self.problem.dimension > 1 and random.random() < self.crossover_rate:
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


class RIGA(GA):
    """
    A Random Immigrants Genetic Algorithm (RIGA) for dynamic optimization.

    This algorithm extends the canonical GA by introducing "random immigrants"
    in each generation to maintain diversity. A fixed percentage of the
    population is replaced by newly generated random individuals, which helps
    the algorithm avoid premature convergence and adapt to changing fitness
    landscapes.

    This implementation is based on the description in Section 3.2.3 and
    Algorithm 3.5 of Pamparà's PhD thesis.
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 immigrant_rate: float,
                 tournament_size: int = 2,
                 **kwargs):
        """
        Initializes the Random Immigrants Genetic Algorithm solver.

        Args:
            problem (Problem[List[float], float]): The dynamic optimization
                problem to solve.
            name (str): the name of the solver
            population_size (int): The number of individuals in the population.
            crossover_rate (float): The probability of crossover.
            mutation_rate (float): The probability of mutation.
            immigrant_rate (float): The proportion of the population to be
                replaced by random immigrants in each generation (p_im).
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
        self.immigrant_rate = immigrant_rate

    def _generate_immigrants(self, num_immigrants: int) -> List[List[float]]:
        """Generates a specified number of new random individuals."""
        immigrants = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(num_immigrants):
            individual = [random.uniform(lower_bounds[i], upper_bounds[i])
                          for i in range(self.problem.dimension)]
            immigrants.append(individual)
        return immigrants

    def step(self) -> None:
        """Performs one generation of the RIGA."""
        # --- Perform a standard GA step first ---
        super().step()

        # --- Introduce Immigrants ---
        num_immigrants = int(self.population_size * self.immigrant_rate)
        if num_immigrants == 0:
            return

        # 1. Generate immigrants
        immigrants = self._generate_immigrants(num_immigrants)
        immigrant_evals = [self.problem.evaluate(ind) for ind in immigrants]

        # 2. Combine with the current population by replacing the worst
        # Find the indices of the `num_immigrants` worst individuals
        sorted_indices = sorted(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        worst_indices = sorted_indices[-num_immigrants:]

        # Replace them with the new immigrants
        for i, idx in enumerate(worst_indices):
            self.population[idx] = immigrants[i]
            self.evaluations[idx] = immigrant_evals[i]
