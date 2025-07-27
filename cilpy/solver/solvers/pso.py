# cilpy/solver/pso.py

import random
from typing import List, Tuple, Any, Optional

from ...problem import Problem
from .. import Solver
from ..chm import ConstraintHandler
from ..chm.debs_rules import DebsRules
from ..chm.no_handler import NoHandler


class GbestPSO(Solver[List[float]]):
    """
    Canonical global best Particle Swarm Optimization (PSO) solver.

    This version is adapted for dynamic optimization problems. It re-evaluates
    personal and global best solutions at the beginning of each step if the
    problem is dynamic, ensuring the swarm's memory is up-to-date with the
    current state of the fitness landscape.
    """

    def __init__(
        self,
        problem: Problem[List[float]],
        swarm_size: int = 30,
        w: float = 0.729,
        c1: float = 2.05,
        c2: float = 2.05,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs: Any
    ):
        """
        Initializes the PSO solver with a problem and algorithm parameters.
        """

        # If no CHT is provided, use DebsRules as a default.
        if constraint_handler is None:
            constraint_handler = NoHandler(problem)

        super().__init__(problem, **kwargs)

        self.swarm_size = swarm_size
        self.w, self.c1, self.c2 = w, c1, c2

        self.chm = constraint_handler

        # --- Initialize Swarm ---
        self.positions = [self.problem.initialize_solution() for _ in range(swarm_size)]
        self.velocities = [self._initialize_velocity() for _ in range(swarm_size)]

        self.pbest_positions = [p.copy() for p in self.positions]
        self.pbest_values = [self.chm.evaluate(pos) for pos in self.pbest_positions]

        # Initialize gbest based on the initial pbest values
        best_idx = 0
        for i in range(1, self.swarm_size):
            if self.chm.is_better(self.pbest_values[i], self.pbest_values[best_idx]):
                best_idx = i
        
        self.gbest_position = self.pbest_positions[best_idx]
        self.gbest_value = self.pbest_values[best_idx]

        self.gbest_idx = min(range(self.swarm_size), key=lambda i: self.pbest_values[i])
        self.gbest_position = self.pbest_positions[self.gbest_idx]
        self.gbest_value = self.pbest_values[self.gbest_idx]

        # Store whether the problem is dynamic to avoid checks every step
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

    def _initialize_velocity(self) -> List[float]:
        lower, upper = self.problem.get_bounds()
        max_velocity = [abs(u - l) * 0.5 for l, u in zip(lower, upper)]
        return [random.uniform(-v, v) for v in max_velocity]

    def _clamp_position(self, position: List[float]) -> List[float]:
        lower, upper = self.problem.get_bounds()
        return [min(max(x, l), u) for x, l, u in zip(position, lower, upper)]

    def step(self) -> None:
        """
        Performs one iteration of the PSO algorithm, updating all particles.
        """
        # Re-evaluate for dynamic problems
        if self.is_dynamic or self.is_constrained_dynamic:
            self.pbest_values = [self.chm.evaluate(pos) for pos in self.pbest_positions]

            self.gbest_value = self.chm.evaluate(self.gbest_position)

            current_best_idx = min(
                range(self.swarm_size), key=lambda i: self.pbest_values[i]
            )
            if self.pbest_values[current_best_idx] < self.gbest_value:
                self.gbest_idx = current_best_idx
                self.gbest_position = self.pbest_positions[self.gbest_idx]
                self.gbest_value = self.pbest_values[self.gbest_idx]

        # The main PSO loop proceeds with up-to-date fitness values for pbest/gbest
        dimension = self.problem.get_dimension()
        for i in range(self.swarm_size):
            # Update velocity
            new_velocity = []
            for d in range(dimension):
                r1, r2 = random.random(), random.random()
                cognitive = (
                    self.c1 * r1 * (self.pbest_positions[i][d] - self.positions[i][d])
                )
                social = self.c2 * r2 * (self.gbest_position[d] - self.positions[i][d])
                velocity = self.w * self.velocities[i][d] + cognitive + social
                new_velocity.append(velocity)
            self.velocities[i] = new_velocity

            # Update position
            new_position = [
                pos + vel for pos, vel in zip(self.positions[i], self.velocities[i])
            ]
            self.positions[i] = self._clamp_position(new_position)

            # Optionally repair the new position using the CHT
            self.positions[i] = self.chm.repair(self.positions[i])

            # Evaluate new position
            new_fitness = self.chm.evaluate(self.positions[i])

            # Update personal best
            if self.chm.is_better(new_fitness, self.pbest_values[i]):
                self.pbest_positions[i] = self.positions[i]
                self.pbest_values[i] = new_fitness

                # Update global best if this new pbest is the best found
                if self.chm.is_better(new_fitness, self.gbest_value):
                    self.gbest_position = self.positions[i]
                    self.gbest_value = new_fitness
                    self.gbest_idx = i

    def get_best(self) -> Tuple[List[float], List[float]]:
        """
        Returns the best solution and its objective value(s) found so far.
        """
        raw_objective = self.problem.get_objective_functions()[0](self.gbest_position)
        return self.gbest_position, [raw_objective]
