# cilpy/solver/solvers/pso.py

import random
import copy
from typing import List, Tuple

from ...problem import Problem, Evaluation, SolutionType, FitnessType
from .. import Solver


class PSO(Solver[List[float], float]):
    """
    A canonical Particle Swarm Optimization (PSO) solver.

    This implementation is based on the algorithm described in Section 3.1.3
    of the reference document, including the inertia weight component. Each
    particle's movement is influenced by its personal best position and the
    swarm's global best position.

    The implementation uses a global best (gbest) topology (star topology).
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 swarm_size: int,
                 w: float,
                 c1: float,
                 c2: float,
                 **kwargs):
        """
        Initializes the Particle Swarm Optimization solver.

        Args:
            problem (Problem[List[float], float]): The optimization problem to solve.
            swarm_size (int): The number of particles in the swarm.
            w (float): The inertia weight, controlling the influence of the
                previous velocity.
            c1 (float): The cognitive coefficient, scaling the influence of the
                particle's personal best.
            c2 (float): The social coefficient, scaling the influence of the
                swarm's global best.
            **kwargs: Additional keyword arguments (not used in this canonical PSO).
        """
        super().__init__(problem, name)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        lower_bounds, upper_bounds = self.problem.bounds

        # Initialize swarm
        self.positions = self._initialize_positions()
        self.velocities = [
            [(random.uniform(-abs(upper_bounds[i] - lower_bounds[i]),
                             abs(upper_bounds[i] - lower_bounds[i])) * 0.1)
             for i in range(self.problem.dimension)]
            for _ in range(self.swarm_size)
        ]

        # Evaluate initial positions and set personal bests
        self.evaluations = [self.problem.evaluate(pos) for pos in self.positions]
        self.pbest_positions = copy.deepcopy(self.positions)
        self.pbest_evaluations = copy.deepcopy(self.evaluations)

        # Initialize global best
        best_initial_idx = min(range(self.swarm_size), key=lambda i: self.evaluations[i].fitness)
        self.gbest_position = copy.deepcopy(self.positions[best_initial_idx])
        self.gbest_evaluation = copy.deepcopy(self.evaluations[best_initial_idx])

    def _initialize_positions(self) -> List[List[float]]:
        """Creates the initial particle positions."""
        positions = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(self.swarm_size):
            pos = [random.uniform(lower_bounds[i], upper_bounds[i])
                   for i in range(self.problem.dimension)]
            positions.append(pos)
        return positions

    def step(self) -> None:
        """Performs one iteration of the PSO algorithm."""
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(self.swarm_size):
            # 1. Update Velocity (Equation 3.1)
            for d in range(self.problem.dimension):
                r1 = random.random()
                r2 = random.random()

                cognitive_component = self.c1 * r1 * (self.pbest_positions[i][d] - self.positions[i][d])
                social_component = self.c2 * r2 * (self.gbest_position[d] - self.positions[i][d])
                inertia_component = self.w * self.velocities[i][d]

                self.velocities[i][d] = inertia_component + cognitive_component + social_component

            # 2. Update Position (Equation 3.2)
            for d in range(self.problem.dimension):
                self.positions[i][d] += self.velocities[i][d]
                # Clamp position to stay within bounds
                self.positions[i][d] = max(lower_bounds[d], min(self.positions[i][d], upper_bounds[d]))

            # 3. Evaluate new position
            self.evaluations[i] = self.problem.evaluate(self.positions[i])

            # 4. Update Personal Best (pbest)
            if self.evaluations[i].fitness < self.pbest_evaluations[i].fitness:
                self.pbest_positions[i] = copy.deepcopy(self.positions[i])
                self.pbest_evaluations[i] = copy.deepcopy(self.evaluations[i])

                # 5. Update Global Best (gbest)
                if self.pbest_evaluations[i].fitness < self.gbest_evaluation.fitness:
                    self.gbest_position = copy.deepcopy(self.pbest_positions[i])
                    self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[i])

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """Returns the global best solution found by the swarm."""
        return [(self.gbest_position, self.gbest_evaluation)]
