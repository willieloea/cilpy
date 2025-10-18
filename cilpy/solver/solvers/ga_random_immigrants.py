# cilpy/solver/solvers/random_immigrants_ga.py

import random
from typing import List, Tuple

from ...problem import Problem, Evaluation
from .ga import GA


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
