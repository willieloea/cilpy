# cilpy/solver/solvers/qpso.py

import random
import numpy as np
from typing import List, Tuple, Any, Optional

from ...problem import Problem
from .. import Solver
from ..chm import ConstraintHandler
from ..chm.debs_rules import DebsRules


def _uniform_distribution(
    local_attractor: float, current_pos: float, mbest_pos: float, alpha: float
) -> float:
    """Position update strategy based on a Uniform distribution."""
    char_length = alpha * abs(mbest_pos - current_pos)
    lower_bound = local_attractor - char_length
    upper_bound = local_attractor + char_length
    return random.uniform(lower_bound, upper_bound)


def _gaussian_distribution(
    local_attractor: float, current_pos: float, mbest_pos: float, alpha: float
) -> float:
    """Position update strategy based on a Gaussian distribution."""
    char_length = alpha * abs(mbest_pos - current_pos)
    # Ensure sigma is not zero to avoid errors with random.gauss
    if char_length <= 1e-9:
        return local_attractor
    return random.gauss(mu=local_attractor, sigma=char_length)


class QPSO(Solver[np.ndarray, np.float64]):
    """
    Quantum-Inspired Particle Swarm Optimization (QPSO) solver.

    This solver implements the QPSO algorithm, which differs from canonical PSO
    by eliminating the velocity vector. Particles are attracted to a stochastic
    local attractor, whose position is influenced by the particle's personal
    best and the global best position.

    This implementation is fully integrated with the cilpy library, supporting
    dynamic and constrained optimization problems through a Constraint Handling
    Mechanism (CHM).
    """

    def __init__(
        self,
        problem: Problem[np.ndarray, np.float64],
        swarm_size: int = 30,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        max_iterations: int = 1000,
        distribution: str = "gaussian",
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs: Any
    ):
        """
        Initializes the QPSO solver.
        """
        if constraint_handler is None:
            constraint_handler = DebsRules(problem)

        super().__init__(problem, **kwargs)

        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer for alpha scheduling.")

        self.swarm_size = swarm_size
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.max_iterations = max_iterations
        self.iteration = 0
        self.chm = constraint_handler

        if distribution.lower() == "uniform":
            self.distribution_strategy = _uniform_distribution
        elif distribution.lower() == "gaussian":
            self.distribution_strategy = _gaussian_distribution
        else:
            raise ValueError("Distribution must be 'uniform' or 'gaussian'.")

        # --- Initialize Swarm using NumPy ---
        lower, upper = self.problem.get_bounds()
        self.positions = np.random.uniform(lower, upper, (self.swarm_size, self.problem.get_dimension()))
        self.pbest_positions = self.positions.copy()
        self.pbest_values = [self.chm.evaluate(pos) for pos in self.pbest_positions]

        # Initialize gbest using the constraint handler's comparison
        gbest_idx = 0
        for i in range(1, self.swarm_size):
            if self.chm.is_better(self.pbest_values[i], self.pbest_values[gbest_idx]):
                gbest_idx = i

        self.gbest_idx = gbest_idx
        self.gbest_position = self.pbest_positions[self.gbest_idx].copy()
        self.gbest_value = self.pbest_values[self.gbest_idx]

        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

    def _clamp_position(self, position: np.ndarray) -> np.ndarray:
        """Clamps a position to the problem's bounds using NumPy."""
        lower, upper = self.problem.get_bounds()
        return np.clip(position, lower, upper)

    def step(self) -> None:
        """Performs one iteration of the QPSO algorithm."""
        # 1. Re-evaluate memory if the environment is dynamic
        if self.is_dynamic or self.is_constrained_dynamic:
            self.pbest_values = [self.chm.evaluate(pos) for pos in self.pbest_positions]
            self.gbest_value = self.chm.evaluate(self.gbest_position)

            current_best_p_idx = 0
            for i in range(1, self.swarm_size):
                if self.chm.is_better(self.pbest_values[i], self.pbest_values[current_best_p_idx]):
                    current_best_p_idx = i
            
            if self.chm.is_better(self.pbest_values[current_best_p_idx], self.gbest_value):
                self.gbest_idx = current_best_p_idx
                self.gbest_position = self.pbest_positions[self.gbest_idx].copy()
                self.gbest_value = self.pbest_values[self.gbest_idx]

        # 2. Calculate the mean best position (mbest) using NumPy
        mbest_pos = np.mean(self.pbest_positions, axis=0)

        # 3. Update alpha (linearly decreasing contraction-expansion coefficient)
        alpha = self.alpha_start - (self.iteration / self.max_iterations) * (
            self.alpha_start - self.alpha_end
        )

        # 4. Update particle positions and evaluate
        for i in range(self.swarm_size):
            phi = np.random.rand(self.problem.get_dimension())
            local_attractor = phi * self.pbest_positions[i] + (1 - phi) * self.gbest_position

            # Use a vectorized approach for applying the distribution strategy
            new_position = np.array([
                self.distribution_strategy(
                    local_attractor[d], self.positions[i, d], mbest_pos[d], alpha
                ) for d in range(self.problem.get_dimension())
            ])

            self.positions[i] = self._clamp_position(new_position)
            self.positions[i] = self.chm.repair(self.positions[i])
            new_fitness = self.chm.evaluate(self.positions[i])

            # Update personal best using the constraint handler
            if self.chm.is_better(new_fitness, self.pbest_values[i]):
                self.pbest_positions[i] = self.positions[i].copy()
                self.pbest_values[i] = new_fitness
                # Update global best
                if self.chm.is_better(new_fitness, self.gbest_value):
                    self.gbest_position = self.positions[i].copy()
                    self.gbest_value = new_fitness
                    self.gbest_idx = i

        self.iteration += 1

    def get_best(self) -> Tuple[np.ndarray, np.float64]:
        """
        Returns the best solution and its raw objective value found so far.
        """
        objective_func = self.problem.get_objective_functions()[0]
        raw_objective_value = objective_func(self.gbest_position)
        return self.gbest_position, raw_objective_value