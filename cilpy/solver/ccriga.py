# cilpy/solver/ccriga.py

import random
from typing import List, Tuple, Any

from ..problem import Problem
from . import Solver

# =============================================================================
# Helper Functions for the Genetic Algorithm
# These are adapted to work for both minimization and maximization.
# =============================================================================


def _tournament_selection(
    population: List[List[float]],
    fitnesses: List[float],
    tournament_size: int,
    maximize: bool = False,
) -> List[float]:
    """
    Selects an individual from the population using tournament selection.
    Can be used for both minimization and maximization.
    """
    if not population:
        raise ValueError("Population cannot be empty for tournament selection.")

    best_participant_idx = -1
    best_fitness = float("-inf") if maximize else float("inf")

    participants_indices = random.sample(range(len(population)), tournament_size)

    for idx in participants_indices:
        current_fitness = fitnesses[idx]
        if (maximize and current_fitness > best_fitness) or (
            not maximize and current_fitness < best_fitness
        ):
            best_fitness = current_fitness
            best_participant_idx = idx

    return population[best_participant_idx]


def _blend_crossover(
    parent1: List[float], parent2: List[float], alpha: float = 0.5
) -> Tuple[List[float], List[float]]:
    """Performs blend crossover (BLX-alpha) on two parents."""
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
    individual: List[float], bounds: List[Tuple[float, float]], p_mutation: float
) -> List[float]:
    """Performs uniform mutation on an individual."""
    mutated_individual = individual[:]

    for i in range(len(mutated_individual)):
        if random.random() < p_mutation:
            mutated_individual[i] = random.uniform(bounds[i][0], bounds[i][1])

    return mutated_individual


# =============================================================================
# Main CCRIGA Solver Class
# =============================================================================


class CCRIGA(Solver[List[float]]):
    """
    Cooperative Co-evolutionary Random Immigrant Genetic Algorithm (CCRIGA).

    This solver adapts the RIGA algorithm to the co-evolutionary framework for
    solving constrained optimization problems. It maintains two populations:
    1.  Solution Population (P1): Searches for the optimal solution vector 'x'.
        It MINIMIZES the Lagrangian function.
    2.  Multiplier Population (P2): Searches for the optimal Lagrangian
        multipliers 'μ' and 'λ'. It MAXIMIZES the Lagrangian function.

    Each population is evolved using a Random Immigrant Genetic Algorithm. The
    fitness of an individual in one population is determined by the best
    individual found so far in the other population.

    The algorithm is designed to handle DCOPs by re-evaluating fitnesses upon
    environmental changes and re-initializing the multiplier population if the
    number of constraints changes.
    """

    def __init__(
        self,
        problem: Problem[List[float]],
        population_size_x: int = 50,
        population_size_l: int = 50,
        p_crossover: float = 0.9,
        p_mutation: float = 0.05,
        p_immigrants: float = 0.1,
        tournament_size: int = 3,
        lambda_bounds: Tuple[float, float] = (0.0, 1000.0),
        **kwargs: Any
    ):
        super().__init__(problem, **kwargs)

        # --- GA Parameters ---
        if population_size_x % 2 != 0:
            population_size_x += 1
        if population_size_l % 2 != 0:
            population_size_l += 1
        self.pop_size_x = population_size_x
        self.pop_size_l = population_size_l
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_immigrants = p_immigrants
        self.tournament_size = tournament_size
        self.num_immigrants_x = int(self.pop_size_x * self.p_immigrants)
        self.num_immigrants_l = int(self.pop_size_l * self.p_immigrants)
        self.lambda_bounds = lambda_bounds

        # --- Problem Details ---
        self.objective = self.problem.get_objective_functions()[0]
        self.bounds_x = self.problem.get_bounds()
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

        # --- Initialize and Evaluate Populations ---
        self._initialize_populations()

    def _initialize_populations(self):
        """Initializes both solution and multiplier populations."""
        # --- Problem Constraints and Multiplier Dimensions ---
        self.inequality_constraints, self.equality_constraints = (
            self.problem.get_constraint_functions()
        )
        self.lambda_dim = len(self.inequality_constraints) + len(
            self.equality_constraints
        )

        # --- Initialize Solution Population (P1 for x) ---
        self.population_x = [
            self.problem.initialize_solution() for _ in range(self.pop_size_x)
        ]

        # --- Initialize Multiplier Population (P2 for μ, λ) ---
        self.gbest_position_l = self._initialize_lambda_individual()  # Placeholder
        if self.lambda_dim > 0:
            self.population_l = [
                self._initialize_lambda_individual() for _ in range(self.pop_size_l)
            ]
            self.bounds_l = [self.lambda_bounds] * self.lambda_dim
        else:
            self.population_l, self.bounds_l = [], []

        # --- Initial Evaluation ---
        # 1. Evaluate P1 using placeholder gbest_l to find an initial gbest_x
        self.fitnesses_x = [
            self._calculate_lagrangian(ind, self.gbest_position_l)
            for ind in self.population_x
        ]
        best_idx_x = min(range(self.pop_size_x), key=lambda i: self.fitnesses_x[i])
        self.gbest_position_x = self.population_x[best_idx_x][:]
        self.gbest_value_x = self.fitnesses_x[best_idx_x]

        # 2. Evaluate P2 using the new gbest_x to find gbest_l
        if self.lambda_dim > 0:
            self.fitnesses_l = [
                self._calculate_lagrangian(self.gbest_position_x, ind)
                for ind in self.population_l
            ]
            best_idx_l = max(range(self.pop_size_l), key=lambda i: self.fitnesses_l[i])
            self.gbest_position_l = self.population_l[best_idx_l][:]
            self.gbest_value_l = self.fitnesses_l[best_idx_l]
        else:
            self.fitnesses_l, self.gbest_position_l, self.gbest_value_l = (
                [],
                [],
                self.gbest_value_x,
            )

    def _initialize_lambda_individual(self) -> List[float]:
        """Creates a single random individual for the multiplier population."""
        if self.lambda_dim == 0:
            return []
        return [random.uniform(*self.lambda_bounds) for _ in range(self.lambda_dim)]

    def _calculate_lagrangian(self, x: List[float], lag_mult: List[float]) -> float:
        """Calculates the value of the Lagrangian function L(x, μ, λ)."""
        obj_val = self.objective(x)
        if not self.lambda_dim:
            return obj_val

        num_ineq = len(self.inequality_constraints)
        mu, lam = lag_mult[:num_ineq], lag_mult[num_ineq:]

        ineq_penalty = sum(
            m * max(0, const(x)) for m, const in zip(mu, self.inequality_constraints)
        )
        eq_penalty = sum(
            l * abs(const(x)) for l, const in zip(lam, self.equality_constraints)
        )
        return obj_val + ineq_penalty + eq_penalty

    def _clamp_position(
        self, position: List[float], bounds: List[Tuple[float, float]]
    ) -> List[float]:
        """Clamps a position to the given bounds."""
        return [max(b[0], min(x, b[1])) for x, b in zip(position, bounds)]

    def _handle_dynamic_change(self):
        """Detects and responds to changes in the dynamic environment."""
        # Check for change in constraint dimensionality
        new_ineq, new_eq = self.problem.get_constraint_functions()
        new_lambda_dim = len(new_ineq) + len(new_eq)

        if new_lambda_dim != self.lambda_dim:
            # Re-initialize both populations if dimensionality changes, as per the paper
            self._initialize_populations()
            return  # Skip the rest of the re-evaluation for this step

        # Re-evaluate all fitnesses using current gbests
        self.fitnesses_x = [
            self._calculate_lagrangian(ind, self.gbest_position_l)
            for ind in self.population_x
        ]
        self.gbest_value_x = self._calculate_lagrangian(
            self.gbest_position_x, self.gbest_position_l
        )

        if self.lambda_dim > 0:
            self.fitnesses_l = [
                self._calculate_lagrangian(self.gbest_position_x, ind)
                for ind in self.population_l
            ]
            self.gbest_value_l = self._calculate_lagrangian(
                self.gbest_position_x, self.gbest_position_l
            )

        # Check for new global bests in the current populations
        current_best_idx_x = min(
            range(self.pop_size_x), key=lambda i: self.fitnesses_x[i]
        )
        if self.fitnesses_x[current_best_idx_x] < self.gbest_value_x:
            self.gbest_position_x = self.population_x[current_best_idx_x][:]
            self.gbest_value_x = self.fitnesses_x[current_best_idx_x]

        if self.lambda_dim > 0:
            current_best_idx_l = max(
                range(self.pop_size_l), key=lambda i: self.fitnesses_l[i]
            )
            if self.fitnesses_l[current_best_idx_l] > self.gbest_value_l:
                self.gbest_position_l = self.population_l[current_best_idx_l][:]
                self.gbest_value_l = self.fitnesses_l[current_best_idx_l]

    def step(self) -> None:
        """Performs one co-evolutionary generation."""
        if self.is_dynamic or self.is_constrained_dynamic:
            self._handle_dynamic_change()

        # --- Evolve Solution Population (P1) - Minimization ---
        offspring_pop_x = []
        for _ in range(self.pop_size_x // 2):
            p1 = _tournament_selection(
                self.population_x, self.fitnesses_x, self.tournament_size
            )
            p2 = _tournament_selection(
                self.population_x, self.fitnesses_x, self.tournament_size
            )
            c1, c2 = (
                _blend_crossover(p1, p2)
                if random.random() < self.p_crossover
                else (p1[:], p2[:])
            )
            c1 = self._clamp_position(
                _uniform_mutation(c1, self.bounds_x, self.p_mutation), self.bounds_x
            )
            c2 = self._clamp_position(
                _uniform_mutation(c2, self.bounds_x, self.p_mutation), self.bounds_x
            )
            offspring_pop_x.extend([c1, c2])

        offspring_fit_x = [
            self._calculate_lagrangian(ind, self.gbest_position_l)
            for ind in offspring_pop_x
        ]

        combined_pop_x = self.population_x + offspring_pop_x
        combined_fit_x = self.fitnesses_x + offspring_fit_x
        sorted_indices = sorted(
            range(len(combined_fit_x)), key=lambda k: combined_fit_x[k]
        )

        self.population_x = [
            combined_pop_x[i] for i in sorted_indices[: self.pop_size_x]
        ]
        self.fitnesses_x = [
            combined_fit_x[i] for i in sorted_indices[: self.pop_size_x]
        ]

        for i in range(self.pop_size_x - self.num_immigrants_x, self.pop_size_x):
            immigrant = self.problem.initialize_solution()
            self.population_x[i] = immigrant
            self.fitnesses_x[i] = self._calculate_lagrangian(
                immigrant, self.gbest_position_l
            )

        if self.fitnesses_x[0] < self.gbest_value_x:
            self.gbest_value_x = self.fitnesses_x[0]
            self.gbest_position_x = self.population_x[0][:]

        # --- Evolve Multiplier Population (P2) - Maximization ---
        if self.lambda_dim > 0:
            offspring_pop_l = []
            for _ in range(self.pop_size_l // 2):
                p1 = _tournament_selection(
                    self.population_l,
                    self.fitnesses_l,
                    self.tournament_size,
                    maximize=True,
                )
                p2 = _tournament_selection(
                    self.population_l,
                    self.fitnesses_l,
                    self.tournament_size,
                    maximize=True,
                )
                c1, c2 = (
                    _blend_crossover(p1, p2)
                    if random.random() < self.p_crossover
                    else (p1[:], p2[:])
                )
                c1 = self._clamp_position(
                    _uniform_mutation(c1, self.bounds_l, self.p_mutation), self.bounds_l
                )
                c2 = self._clamp_position(
                    _uniform_mutation(c2, self.bounds_l, self.p_mutation), self.bounds_l
                )
                offspring_pop_l.extend([c1, c2])

            offspring_fit_l = [
                self._calculate_lagrangian(self.gbest_position_x, ind)
                for ind in offspring_pop_l
            ]

            combined_pop_l = self.population_l + offspring_pop_l
            combined_fit_l = self.fitnesses_l + offspring_fit_l
            sorted_indices = sorted(
                range(len(combined_fit_l)),
                key=lambda k: combined_fit_l[k],
                reverse=True,
            )  # Maximize

            self.population_l = [
                combined_pop_l[i] for i in sorted_indices[: self.pop_size_l]
            ]
            self.fitnesses_l = [
                combined_fit_l[i] for i in sorted_indices[: self.pop_size_l]
            ]

            for i in range(self.pop_size_l - self.num_immigrants_l, self.pop_size_l):
                immigrant = self._initialize_lambda_individual()
                self.population_l[i] = immigrant
                self.fitnesses_l[i] = self._calculate_lagrangian(
                    self.gbest_position_x, immigrant
                )

            if self.fitnesses_l[0] > self.gbest_value_l:
                self.gbest_value_l = self.fitnesses_l[0]
                self.gbest_position_l = self.population_l[0][:]

    def get_best(self) -> Tuple[List[float], List[float]]:
        """Returns the best solution 'x' and its raw objective value."""
        best_objective_value = self.objective(self.gbest_position_x)
        return self.gbest_position_x, [best_objective_value]
