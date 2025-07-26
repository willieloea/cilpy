# cilpy/solver/gvpso.py

import random
from typing import List, Tuple, Any

from ...cilpy.problem import Problem
from ...cilpy.solver import Solver

# =============================================================================
# Main Gaussian-Valued Particle Swarm Optimisation Solver Class
# =============================================================================


class GVPSOSolver(Solver[List[float]]):
    """
    Gaussian-Valued Particle Swarm Optimization (GVPSO) solver.

    This solver implements the GVPSO algorithm as proposed by Harrison. GVPSO is
    a variant of PSO that updates particle positions by sampling from a Gaussian
    distribution, aiming to maintain the "PSO identity" while being more
    parameter-lite, similar to Bare-Bones PSO.

    The algorithm's core mechanics involve an "ancillary position" and a
    probabilistic update rule that either exploits the particle's personal
    best position or explores a new position sampled from a Gaussian
    distribution.

    This implementation is adapted for dynamic optimization problems by
    re-evaluating memory (pbest/gbest) at the start of each step.

    Referrences:
        K. R. Harrison. “An Analysis of Parameter Control Mechanisms for the
        Particle Swarm Optimization Algorithm”.
    """

    def __init__(
        self,
        problem: Problem[List[float]],
        swarm_size: int = 30,
        e: float = 0.5,
        **kwargs: Any
    ):
        """
        Initializes the GVPSO solver.

        Args:
            problem: The optimization problem to solve.
            swarm_size: The number of particles in the swarm.
            e: A user-defined parameter controlling the degree to which the
               personal best position is exploited. A value between 0 and 1.
            **kwargs: Additional parameters (ignored).
        """
        super().__init__(problem, **kwargs)
        if not (0 <= e <= 1):
            raise ValueError("'e' parameter must be between 0 and 1.")

        self.swarm_size = swarm_size
        self.e = e
        self.objective = self.problem.get_objective_functions()[0]
        self.dimension = self.problem.get_dimension()

        # Initialize particles and global best
        self.positions = [
            self.problem.initialize_solution() for _ in range(self.swarm_size)
        ]
        self.pbest_positions = [p.copy() for p in self.positions]
        self.pbest_values = [self.objective(pos) for pos in self.pbest_positions]

        gbest_idx = min(range(self.swarm_size), key=lambda i: self.pbest_values[i])
        self.gbest_position = self.pbest_positions[gbest_idx].copy()
        self.gbest_value = self.pbest_values[gbest_idx]

        # Store dynamic status to avoid repeated checks
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

    def _clamp_position(self, position: List[float]) -> List[float]:
        """Clamps a position to the problem's bounds."""
        lower, upper = self.problem.get_bounds()
        return [max(l, min(x, u)) for x, l, u in zip(position, lower, upper)]

    def step(self) -> None:
        """Performs one iteration of the GVPSO algorithm."""

        # Re-evaluate memory if the environment is dynamic
        if self.is_dynamic or self.is_constrained_dynamic:
            self.pbest_values = [self.objective(pos) for pos in self.pbest_positions]
            self.gbest_value = self.objective(self.gbest_position)

            current_best_idx = min(
                range(self.swarm_size), key=lambda i: self.pbest_values[i]
            )
            if self.pbest_values[current_best_idx] < self.gbest_value:
                self.gbest_position = self.pbest_positions[current_best_idx].copy()
                self.gbest_value = self.pbest_values[current_best_idx]

        # Update particle positions
        for i in range(self.swarm_size):
            new_position = []
            for d in range(self.dimension):
                # Check for exploitation of personal best (y_ij)
                if random.random() < self.e:
                    new_pos_d = self.pbest_positions[i][d]
                else:
                    # Exploration using Gaussian sampling
                    current_pos_d = self.positions[i][d]
                    pbest_pos_d = self.pbest_positions[i][d]
                    gbest_pos_d = self.gbest_position[d]

                    # Calculate ancillary position component (Δ_ij) per eq (3.5)
                    r1 = random.random()
                    r2 = random.random()
                    ancillary_pos_d = (
                        current_pos_d
                        + r1 * (pbest_pos_d - current_pos_d)
                        + r2 * (gbest_pos_d - current_pos_d)
                    )

                    # Calculate Gaussian parameters per eq (3.4)
                    mean = (current_pos_d + ancillary_pos_d) / 2.0
                    std_dev = abs(ancillary_pos_d - current_pos_d)

                    # Sample from Gaussian distribution
                    # Handle case where standard deviation is zero to avoid errors
                    if std_dev < 1e-9:
                        new_pos_d = mean
                    else:
                        new_pos_d = random.gauss(mean, std_dev)

                new_position.append(new_pos_d)

            # Update, clamp, and evaluate the new position
            self.positions[i] = self._clamp_position(new_position)
            new_fitness = self.objective(self.positions[i])

            # Update personal and global bests
            if new_fitness < self.pbest_values[i]:
                self.pbest_positions[i] = self.positions[i].copy()
                self.pbest_values[i] = new_fitness

                if new_fitness < self.gbest_value:
                    self.gbest_position = self.positions[i].copy()
                    self.gbest_value = new_fitness

    def get_best(self) -> Tuple[List[float], List[float]]:
        """
        Returns the best solution and its corresponding objective value found so far.

        Returns:
            A tuple containing the best solution (List[float]) and its
            objective value (wrapped in a List[float]).
        """
        return self.gbest_position, [self.gbest_value]
