# cilpy/solver/solvers/ccriga.py

import random
import numpy as np
from typing import List, Tuple, Any

from ...problem import Problem
from .. import Solver

# =============================================================================
# Helper Functions for the Genetic Algorithm (NumPy Version)
# =============================================================================

def _tournament_selection(
    population: np.ndarray,
    fitnesses: np.ndarray,
    tournament_size: int,
    maximize: bool = False,
) -> np.ndarray:
    """Selects an individual using tournament selection from a NumPy population."""
    if population.shape[0] == 0:
        raise ValueError("Population cannot be empty for tournament selection.")

    participants_indices = np.random.choice(population.shape[0], tournament_size, replace=False)
    
    if maximize:
        best_local_idx = np.argmax(fitnesses[participants_indices])
    else:
        best_local_idx = np.argmin(fitnesses[participants_indices])
        
    best_global_idx = participants_indices[best_local_idx]
    
    return population[best_global_idx]


def _blend_crossover(
    parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs blend crossover (BLX-alpha) on two NumPy array parents."""
    d = np.abs(parent1 - parent2)
    lower = np.minimum(parent1, parent2) - alpha * d
    upper = np.maximum(parent1, parent2) + alpha * d
    
    child1 = np.random.uniform(lower, upper)
    child2 = np.random.uniform(lower, upper)
    
    return child1, child2


def _uniform_mutation(
    individual: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], p_mutation: float
) -> np.ndarray:
    """Performs uniform mutation on a NumPy array individual."""
    mutated_individual = individual.copy()
    mutation_mask = np.random.random(size=individual.shape[0]) < p_mutation
    
    if np.any(mutation_mask):
        lower_b, upper_b = bounds
        mutated_individual[mutation_mask] = np.random.uniform(
            lower_b[mutation_mask], upper_b[mutation_mask]
        )
            
    return mutated_individual


# =============================================================================
# Main CCRIGA Solver Class
# =============================================================================

class CCRIGA(Solver[np.ndarray, float]):
    """
    Cooperative Co-evolutionary Random Immigrant Genetic Algorithm (CCRIGA).

    This solver adapts RIGA to the co-evolutionary framework for constrained
    optimization. It maintains two populations:
    1.  Solution Population (P1): Searches 'x', MINIMIZING the Lagrangian function.
    2.  Multiplier Population (P2): Searches 'μ' and 'λ', MAXIMIZING the Lagrangian.

    Each population evolves via a GA with random immigrants. It is adapted for
    DCOPs by re-evaluating fitnesses upon environmental changes.
    """

    def __init__(
        self,
        problem: Problem[np.ndarray, float],
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
        self.pop_size_x = population_size_x + (population_size_x % 2)
        self.pop_size_l = population_size_l + (population_size_l % 2)
        self.p_crossover, self.p_mutation = p_crossover, p_mutation
        self.tournament_size = tournament_size
        self.num_immigrants_x = int(self.pop_size_x * p_immigrants)
        self.num_immigrants_l = int(self.pop_size_l * p_immigrants)
        self.lambda_bounds = lambda_bounds

        # --- Problem Details ---
        self.objective = self.problem.get_objective_functions()[0]
        self.bounds_x = self.problem.get_bounds()
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

        # --- Initialize and Evaluate Populations ---
        self._initialize_populations()

    def _initialize_populations(self):
        """Initializes both solution and multiplier populations using NumPy."""
        self.ineq_constraints, self.eq_constraints = self.problem.get_constraint_functions()
        self.lambda_dim = len(self.ineq_constraints) + len(self.eq_constraints)

        # --- P1: Solution Population (x) ---
        self.population_x = np.array([self._initialize_solution_individual() for _ in range(self.pop_size_x)])
        
        # --- P2: Multiplier Population (μ, λ) ---
        self.gbest_position_l = self._initialize_lambda_individual() # Placeholder
        if self.lambda_dim > 0:
            self.population_l = np.array([self._initialize_lambda_individual() for _ in range(self.pop_size_l)])
            self.bounds_l = (
                np.full(self.lambda_dim, self.lambda_bounds[0]),
                np.full(self.lambda_dim, self.lambda_bounds[1])
            )
        else:
            self.population_l = np.array([])
            self.bounds_l = (np.array([]), np.array([]))

        # --- Initial Evaluation ---
        self._full_evaluation()

    def _full_evaluation(self):
        """Evaluates all populations and finds the initial global bests."""
        # 1. Evaluate P1 using placeholder/current gbest_l
        self.fitnesses_x = np.array([self._calculate_lagrangian(ind, self.gbest_position_l) for ind in self.population_x])
        best_idx_x = np.argmin(self.fitnesses_x)
        self.gbest_position_x = self.population_x[best_idx_x].copy()
        self.gbest_value_x = self.fitnesses_x[best_idx_x]

        # 2. Evaluate P2 using the new gbest_x
        if self.lambda_dim > 0:
            self.fitnesses_l = np.array([self._calculate_lagrangian(self.gbest_position_x, ind) for ind in self.population_l])
            best_idx_l = np.argmax(self.fitnesses_l)
            self.gbest_position_l = self.population_l[best_idx_l].copy()
            self.gbest_value_l = self.fitnesses_l[best_idx_l]
        else:
            self.fitnesses_l, self.gbest_value_l = np.array([]), self.gbest_value_x

    def _initialize_solution_individual(self) -> np.ndarray:
        """Creates a single random individual for the solution population."""
        dim = self.problem.get_dimension()
        lower_b, upper_b = self.bounds_x
        return np.random.uniform(lower_b, upper_b, size=dim)
        
    def _initialize_lambda_individual(self) -> np.ndarray:
        """Creates a single random individual for the multiplier population."""
        if self.lambda_dim == 0:
            return np.array([])
        return np.random.uniform(self.lambda_bounds[0], self.lambda_bounds[1], size=self.lambda_dim)

    def _calculate_lagrangian(self, x: np.ndarray, lag_mult: np.ndarray) -> float:
        obj_val = self.objective(x)
        if self.lambda_dim == 0:
            return obj_val

        num_ineq = len(self.ineq_constraints)
        mu, lam = lag_mult[:num_ineq], lag_mult[num_ineq:]
        
        ineq_penalty = sum(m * max(0, const(x)) for m, const in zip(mu, self.ineq_constraints))
        eq_penalty = sum(l * abs(const(x)) for l, const in zip(lam, self.eq_constraints))
        return obj_val + ineq_penalty + eq_penalty

    def _clamp_position(self, position: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Clamps a position to the given bounds using np.clip."""
        return np.clip(position, bounds[0], bounds[1])

    def _handle_dynamic_change(self):
        """Re-evaluates populations after a dynamic change."""
        self._full_evaluation()

    def step(self) -> None:
        """Performs one co-evolutionary generation."""
        if self.is_dynamic or self.is_constrained_dynamic:
            self._handle_dynamic_change()

        # --- Evolve Solution Population (P1) - Minimization ---
        self._evolve_population_x()

        # --- Evolve Multiplier Population (P2) - Maximization ---
        if self.lambda_dim > 0:
            self._evolve_population_l()

    def _evolve_population_x(self):
        """Evolves the solution population for one generation."""
        offspring_pop_x = []
        for _ in range(self.pop_size_x // 2):
            p1 = _tournament_selection(self.population_x, self.fitnesses_x, self.tournament_size)
            p2 = _tournament_selection(self.population_x, self.fitnesses_x, self.tournament_size)
            if random.random() < self.p_crossover:
                c1, c2 = _blend_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            c1 = _uniform_mutation(c1, self.bounds_x, self.p_mutation)
            c2 = _uniform_mutation(c2, self.bounds_x, self.p_mutation)

            offspring_pop_x.extend([self._clamp_position(c1, self.bounds_x), self._clamp_position(c2, self.bounds_x)])
        
        offspring_pop_x = np.array(offspring_pop_x)
        offspring_fit_x = np.array([self._calculate_lagrangian(ind, self.gbest_position_l) for ind in offspring_pop_x])

        combined_pop = np.vstack([self.population_x, offspring_pop_x])
        combined_fit = np.concatenate([self.fitnesses_x, offspring_fit_x])
        
        # Elitism: select the best individuals
        sorted_indices = np.argsort(combined_fit)
        survivor_indices = sorted_indices[:self.pop_size_x - self.num_immigrants_x]
        
        self.population_x[:len(survivor_indices)] = combined_pop[survivor_indices]
        self.fitnesses_x[:len(survivor_indices)] = combined_fit[survivor_indices]

        # Replace the worst with random immigrants
        for i in range(self.pop_size_x - self.num_immigrants_x, self.pop_size_x):
            immigrant = self._initialize_solution_individual()
            self.population_x[i] = immigrant
            self.fitnesses_x[i] = self._calculate_lagrangian(immigrant, self.gbest_position_l)

        # Update global best after immigration
        best_idx = np.argmin(self.fitnesses_x)
        if self.fitnesses_x[best_idx] < self.gbest_value_x:
            self.gbest_value_x = self.fitnesses_x[best_idx]
            self.gbest_position_x = self.population_x[best_idx].copy()
    
    def _evolve_population_l(self):
        """Evolves the multiplier population for one generation."""
        offspring_pop_l = []
        for _ in range(self.pop_size_l // 2):
            p1 = _tournament_selection(self.population_l, self.fitnesses_l, self.tournament_size, maximize=True)
            p2 = _tournament_selection(self.population_l, self.fitnesses_l, self.tournament_size, maximize=True)
            if random.random() < self.p_crossover:
                c1, c2 = _blend_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = _uniform_mutation(c1, self.bounds_l, self.p_mutation)
            c2 = _uniform_mutation(c2, self.bounds_l, self.p_mutation)
            
            offspring_pop_l.extend([self._clamp_position(c1, self.bounds_l), self._clamp_position(c2, self.bounds_l)])
        
        offspring_pop_l = np.array(offspring_pop_l)
        offspring_fit_l = np.array([self._calculate_lagrangian(self.gbest_position_x, ind) for ind in offspring_pop_l])

        combined_pop = np.vstack([self.population_l, offspring_pop_l])
        combined_fit = np.concatenate([self.fitnesses_l, offspring_fit_l])

        # Elitism (maximization)
        sorted_indices = np.argsort(combined_fit)[::-1]
        survivor_indices = sorted_indices[:self.pop_size_l - self.num_immigrants_l]
        
        self.population_l[:len(survivor_indices)] = combined_pop[survivor_indices]
        self.fitnesses_l[:len(survivor_indices)] = combined_fit[survivor_indices]

        # Replace the worst with random immigrants
        for i in range(self.pop_size_l - self.num_immigrants_l, self.pop_size_l):
            immigrant = self._initialize_lambda_individual()
            self.population_l[i] = immigrant
            self.fitnesses_l[i] = self._calculate_lagrangian(self.gbest_position_x, immigrant)
        
        # Update global best after immigration
        best_idx = np.argmax(self.fitnesses_l)
        if self.fitnesses_l[best_idx] > self.gbest_value_l:
            self.gbest_value_l = self.fitnesses_l[best_idx]
            self.gbest_position_l = self.population_l[best_idx].copy()

    def get_best(self) -> Tuple[np.ndarray, float]:
        """Returns the best solution 'x' and its raw objective value."""
        best_objective_value = self.objective(self.gbest_position_x)
        return self.gbest_position_x, float(best_objective_value)