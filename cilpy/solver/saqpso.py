# cilpy/solver/saqpso.py

import random
import math
from typing import List, Tuple, Any

from ..problem import Problem
from . import Solver

class SaQPSOSolver(Solver[List[float]]):
    """
    Self-Adaptive Quantum-Inspired Particle Swarm Optimization (SaQPSO) solver.

    This solver implements the SaQPSO algorithm. It divides the swarm into two
    subgroups:
    1. Neutral particles (Sn): Updated using standard PSO velocity and position
       equations.
    2. Quantum particles (Sq): Updated using a QPSO position update rule.

    The key feature is the self-adaptation of the quantum cloud radius,
    `rcloud`.
    This parameter is dynamically calculated at each iteration based on the
    diversity of the two subgroups, eliminating the need for a manually tuned,
    static radius. The diversity is defined as the average Euclidean distance of
    particles in a subgroup from their collective centroid.
    """

    def __init__(self,
                 problem: Problem[List[float]],
                 swarm_size: int = 30,
                 neutral_ratio: float = 0.5,
                 w: float = 0.729,
                 c1: float = 1.494,
                 c2: float = 1.494,
                 **kwargs: Any):
        """
        Initializes the SaQPSO solver.

        Args:
            problem: The optimization problem to solve.
            swarm_size: Total number of particles in the swarm.
            neutral_ratio: The proportion of the swarm to be designated as
                           neutral particles (updated with PSO rules). The rest
                           will be quantum particles. Defaults to 0.5 (50%).
            w: Inertia weight for neutral particles.
            c1: Cognitive coefficient for neutral particles.
            c2: Social coefficient for neutral particles.
            **kwargs: Additional parameters (ignored).
        """
        super().__init__(problem, **kwargs)
        if not (0 < neutral_ratio < 1):
            raise ValueError("neutral_ratio must be between 0 and 1.")

        self.swarm_size = swarm_size
        self.neutral_ratio = neutral_ratio
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.iteration = 0
        self.objective = self.problem.get_objective_functions()[0]
        self.dimension = self.problem.get_dimension()
        
        # --- Swarm Partitioning ---
        self.num_neutral = int(self.swarm_size * self.neutral_ratio)
        self.num_quantum = self.swarm_size - self.num_neutral
        self.neutral_indices = list(range(self.num_neutral))
        self.quantum_indices = list(range(self.num_neutral, self.swarm_size))
        
        if self.num_neutral == 0 or self.num_quantum == 0:
            raise ValueError("Both neutral and quantum subgroups must have at least one particle.")

        # --- Initialize particles and memory ---
        self.positions = [self.problem.initialize_solution() for _ in range(self.swarm_size)]
        self.pbest_positions = [p.copy() for p in self.positions]
        self.pbest_values = [self.objective(pos) for pos in self.pbest_positions]
        
        # Neutral particle velocities (only for the neutral subgroup)
        self.velocities = [[0.0] * self.dimension for _ in range(self.num_neutral)]
        
        # --- Global Best Initialization ---
        gbest_idx = min(range(self.swarm_size), key=lambda i: self.pbest_values[i])
        self.gbest_position = self.pbest_positions[gbest_idx]
        self.gbest_value = self.pbest_values[gbest_idx]
        
        # --- Self-Adaptive Parameter Initialization ---
        self.rcloud = 0.0
        self._update_rcloud() # Initial rcloud calculation
        
        # Store dynamic status to avoid repeated checks
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

    def _clamp_position(self, position: List[float]) -> List[float]:
        """Clamps a position to the problem's bounds."""
        lower, upper = self.problem.get_bounds()
        return [max(l, min(x, u)) for x, l, u in zip(position, lower, upper)]

    def _calculate_diversity(self, particle_indices: List[int]) -> float:
        """
        Calculates the diversity of a particle subgroup.
        Diversity is the average Euclidean distance from the subgroup's centroid.
        """
        subgroup_size = len(particle_indices)
        if subgroup_size == 0:
            return 0.0

        # Calculate centroid (mean position) of the subgroup
        centroid = [0.0] * self.dimension
        for i in particle_indices:
            for d in range(self.dimension):
                centroid[d] += self.positions[i][d]
        for d in range(self.dimension):
            centroid[d] /= subgroup_size

        # Calculate average Euclidean distance from the centroid
        total_distance = 0.0
        for i in particle_indices:
            sq_distance = 0.0
            for d in range(self.dimension):
                sq_distance += (self.positions[i][d] - centroid[d]) ** 2
            total_distance += math.sqrt(sq_distance)
            
        return total_distance / subgroup_size

    def _update_rcloud(self):
        """
        Updates the quantum cloud radius `rcloud`.
        """
        diversity_neutral = self._calculate_diversity(self.neutral_indices)
        diversity_quantum = self._calculate_diversity(self.quantum_indices)
        self.rcloud = max(diversity_neutral, diversity_quantum)

    def step(self) -> None:
        """Performs one iteration of the SaQPSO algorithm."""
        
        # Re-evaluate memory if the environment is dynamic
        if self.is_dynamic or self.is_constrained_dynamic:
            self.pbest_values = [self.objective(pos) for pos in self.pbest_positions]
            self.gbest_value = self.objective(self.gbest_position)
            
            current_best_idx = min(range(self.swarm_size), key=lambda i: self.pbest_values[i])
            if self.pbest_values[current_best_idx] < self.gbest_value:
                self.gbest_position = self.pbest_positions[current_best_idx]
                self.gbest_value = self.pbest_values[current_best_idx]

        # Update Neutral Particles (Sn) using PSO rules
        for i in self.neutral_indices:
            new_velocity = [0.0] * self.dimension
            for d in range(self.dimension):
                r1, r2 = random.random(), random.random()
                cognitive_comp = self.c1 * r1 * (self.pbest_positions[i][d] - self.positions[i][d])
                social_comp = self.c2 * r2 * (self.gbest_position[d] - self.positions[i][d])
                new_velocity[d] = self.w * self.velocities[i][d] + cognitive_comp + social_comp
            
            self.velocities[i] = new_velocity
            
            new_position = [self.positions[i][d] + self.velocities[i][d] for d in range(self.dimension)]
            self.positions[i] = self._clamp_position(new_position)

            # Evaluate and update memory
            new_fitness = self.objective(self.positions[i])
            if new_fitness < self.pbest_values[i]:
                self.pbest_values[i] = new_fitness
                self.pbest_positions[i] = self.positions[i]
                if new_fitness < self.gbest_value:
                    self.gbest_value = new_fitness
                    self.gbest_position = self.positions[i]

        # Update Quantum Particles (Sq) using QPSO rules
        for i in self.quantum_indices:
            new_position = []
            for d in range(self.dimension):
                phi = random.random()
                local_attractor = phi * self.pbest_positions[i][d] + (1 - phi) * self.gbest_position[d]
                
                # Update position using a uniform distribution around the local attractor
                # with a spread determined by the adaptive rcloud.
                # This corresponds to a bounded distribution.
                u_bound = local_attractor + self.rcloud
                l_bound = local_attractor - self.rcloud
                new_position.append(random.uniform(l_bound, u_bound))

            self.positions[i] = self._clamp_position(new_position)

            # Evaluate and update memory
            new_fitness = self.objective(self.positions[i])
            if new_fitness < self.pbest_values[i]:
                self.pbest_values[i] = new_fitness
                self.pbest_positions[i] = self.positions[i]
                if new_fitness < self.gbest_value:
                    self.gbest_value = new_fitness
                    self.gbest_position = self.positions[i]

        # Update rcloud for the next iteration
        self._update_rcloud()
        self.iteration += 1

    def get_best(self) -> Tuple[List[float], List[float]]:
        """Returns the best solution and its objective value found so far."""
        return self.gbest_position, [self.gbest_value]