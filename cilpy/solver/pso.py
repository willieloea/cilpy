from typing import List, Tuple
from ..problem import Problem
from . import Solver
import random

class GbestPSO(Solver[List[float]]):
    """
    Canonical global best Particle Swarm Optimization (PSO) solver.

    Implements the Solver interface for problems with List[float] solutions.
    """
    def __init__(self, problem: Problem[List[float]], swarm_size: int = 30, 
                 w: float = 0.729, c1: float = 2.05, c2: float = 2.05, **kwargs):
        """
        Initializes the PSO solver with a problem and algorithm parameters.

        Args:
            problem (Problem[List[float]]): The optimization problem to solve.
            swarm_size (int): Number of particles in the swarm (default: 30).
            w (float): Inertia weight (default: 0.729, from Clerc & Kennedy, 2002).
            c1 (float): Cognitive coefficient (default: 2.05).
            c2 (float): Social coefficient (default: 2.05).
            **kwargs: Additional parameters (ignored for now).
        """
        super().__init__(problem)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Initialize particles, velocities, personal bests, and global best
        self.positions = [self.problem.initialize_solution() for _ in range(swarm_size)]
        self.velocities = [self._initialize_velocity() for _ in range(swarm_size)]
        self.pbest_positions = self.positions.copy()
        self.pbest_values = [self.problem.evaluate(pos)[0] for pos in self.positions]
        self.gbest_idx = min(range(swarm_size), key=lambda i: self.pbest_values[i][0])
        self.gbest_position = self.pbest_positions[self.gbest_idx]
        self.gbest_value = self.pbest_values[self.gbest_idx]

    def _initialize_velocity(self) -> List[float]:
        """
        Initializes a particle's velocity randomly within bounds-derived limits.

        Returns:
            List[float]: Initial velocity vector.
        """
        lower, upper = self.problem.get_bounds()
        dimension = self.problem.get_dimension()
        max_velocity = [abs(u - l) * 0.5 for l, u in zip(lower, upper)]  # Half the bound range
        return [random.uniform(-v, v) for v in max_velocity]

    def _clamp_position(self, position: List[float]) -> List[float]:
        """
        Clamps a position to the problem's bounds.

        Args:
            position (List[float]): The position to clamp.

        Returns:
            List[float]: The clamped position.
        """
        lower, upper = self.problem.get_bounds()
        return [min(max(x, l), u) for x, l, u in zip(position, lower, upper)]

    def step(self) -> None:
        """
        Performs one iteration of the PSO algorithm, updating all particles.
        """
        dimension = self.problem.get_dimension()

        for i in range(self.swarm_size):
            # Update velocity
            new_velocity = []
            for d in range(dimension):
                r1 = random.random()
                r2 = random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i][d] - self.positions[i][d])
                social = self.c2 * r2 * (self.gbest_position[d] - self.positions[i][d])
                velocity = self.w * self.velocities[i][d] + cognitive + social
                new_velocity.append(velocity)

            # Update position
            new_position = [self.positions[i][d] + new_velocity[d] for d in range(dimension)]
            new_position = self._clamp_position(new_position)

            # Evaluate new position
            objectives, constraints = self.problem.evaluate(new_position)
            is_feasible = all(c <= 0 for c in constraints)  # Feasible if no violations

            # Update personal best
            if is_feasible and (not self.pbest_values[i] or objectives[0] < self.pbest_values[i][0]):
                self.pbest_positions[i] = new_position
                self.pbest_values[i] = objectives

            # Update global best
            if is_feasible and objectives[0] < self.gbest_value[0]:
                self.gbest_position = new_position
                self.gbest_value = objectives
                self.gbest_idx = i

            # Update particle
            self.positions[i] = new_position
            self.velocities[i] = new_velocity

        # Update problem state for dynamic problems
        if any(self.problem.is_dynamic()):
            self.problem.update(iteration=1)  # Iteration count can be tracked externally

    def get_best(self) -> Tuple[List[float], List[float]]:
        """
        Returns the best solution and its objective value(s) found so far.

        Returns:
            Tuple[List[float], List[float]]: The global best position and its objective value(s).
        """
        return self.gbest_position, self.gbest_value