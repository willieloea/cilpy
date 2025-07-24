# cilpy/solver/qpso.py

import random
from typing import List, Tuple, Any

from ..problem import Problem
from . import Solver


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


# =============================================================================
# Main QPSO Solver Class
# =============================================================================


class QPSOSolver(Solver[List[float]]):
    """
    Quantum-Inspired Particle Swarm Optimization (QPSO) solver.

    This solver implements the QPSO algorithm, which differs from canonical PSO
    by eliminating the velocity vector. Instead, particles are attracted to a
    stochastic point within the problem space.

    This implementation is adapted for dynamic optimization problems by
    re-evaluating memory (pbest/gbest) at the start of each step.
    """

    def __init__(
        self,
        problem: Problem[List[float]],
        swarm_size: int = 30,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        max_iterations: int = 1000,
        distribution: str = "uniform",
        **kwargs: Any
    ):
        """
        Initializes the QPSO solver.

        Args:
            problem: The optimization problem to solve.
            swarm_size: Number of particles in the swarm.
            alpha_start: Initial value for the contraction-expansion coefficient.
            alpha_end: Final value for the contraction-expansion coefficient.
            max_iterations: The total number of iterations for the run. This is
                            required to schedule the linear decrease of alpha.
            distribution: The sampling strategy for updating positions.
                          Can be 'uniform' or 'gaussian'.
            **kwargs: Additional parameters (ignored).
        """
        super().__init__(problem, **kwargs)
        self.swarm_size = swarm_size
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.max_iterations = max_iterations
        self.iteration = 0
        self.objective = self.problem.get_objective_functions()[0]
        self.dimension = self.problem.get_dimension()

        # Set the distribution strategy
        if distribution.lower() == "uniform":
            self.distribution_strategy = _uniform_distribution
        elif distribution.lower() == "gaussian":
            self.distribution_strategy = _gaussian_distribution
        else:
            raise ValueError("Distribution must be 'uniform' or 'gaussian'.")

        # Initialize particles and global best
        self.positions = [
            self.problem.initialize_solution() for _ in range(self.swarm_size)
        ]
        self.pbest_positions = [p.copy() for p in self.positions]
        self.pbest_values = [self.objective(pos) for pos in self.pbest_positions]

        self.gbest_idx = min(range(self.swarm_size), key=lambda i: self.pbest_values[i])
        self.gbest_position = self.pbest_positions[self.gbest_idx]
        self.gbest_value = self.pbest_values[self.gbest_idx]

        # Store dynamic status to avoid repeated checks
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

    def _clamp_position(self, position: List[float]) -> List[float]:
        """Clamps a position to the problem's bounds."""
        lower, upper = self.problem.get_bounds()
        return [max(l, min(x, u)) for x, l, u in zip(position, lower, upper)]

    def step(self) -> None:
        """Performs one iteration of the QPSO algorithm."""

        # 1. Re-evaluate memory if the environment is dynamic
        if self.is_dynamic or self.is_constrained_dynamic:
            self.pbest_values = [self.objective(pos) for pos in self.pbest_positions]
            self.gbest_value = self.objective(self.gbest_position)

            current_best_idx = min(
                range(self.swarm_size), key=lambda i: self.pbest_values[i]
            )
            if self.pbest_values[current_best_idx] < self.gbest_value:
                self.gbest_idx = current_best_idx
                self.gbest_position = self.pbest_positions[self.gbest_idx]
                self.gbest_value = self.pbest_values[self.gbest_idx]

        # 2. Calculate the mean best position (mbest)
        mbest_pos = [
            sum(p[d] for p in self.pbest_positions) / self.swarm_size
            for d in range(self.dimension)
        ]

        # 3. Update alpha (linearly decreasing contraction-expansion coefficient)
        alpha = self.alpha_start - (self.iteration / self.max_iterations) * (
            self.alpha_start - self.alpha_end
        )

        # 4. Update particle positions and evaluate
        for i in range(self.swarm_size):
            new_position = []
            for d in range(self.dimension):
                # Calculate the local attractor for each dimension
                phi = random.random()
                local_attractor = (
                    phi * self.pbest_positions[i][d]
                    + (1 - phi) * self.gbest_position[d]
                )

                # Calculate the new position for this dimension
                new_pos_d = self.distribution_strategy(
                    local_attractor, self.positions[i][d], mbest_pos[d], alpha
                )
                new_position.append(new_pos_d)

            # Update and clamp the full position vector
            self.positions[i] = self._clamp_position(new_position)

            # Evaluate new position
            new_fitness = self.objective(self.positions[i])

            # Update personal best
            if new_fitness < self.pbest_values[i]:
                self.pbest_positions[i] = self.positions[i]
                self.pbest_values[i] = new_fitness

                # Update global best
                if new_fitness < self.gbest_value:
                    self.gbest_position = self.positions[i]
                    self.gbest_value = new_fitness
                    self.gbest_idx = i

        self.iteration += 1

    def get_best(self) -> Tuple[List[float], List[float]]:
        """Returns the best solution and its objective value found so far."""
        return self.gbest_position, [self.gbest_value]
