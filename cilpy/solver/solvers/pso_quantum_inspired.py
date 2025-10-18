# cilpy/solver/solvers/pso_quantum_inspired.py

import random
import copy
from typing import List, Tuple

from ...problem import Problem, Evaluation
from .pso import PSO


class QPSO(PSO):
    """
    A Quantum Particle Swarm Optimization (QPSO) algorithm for dynamic problems.

    QPSO enhances diversity by splitting the swarm into two subgroups:
    1.  Neutral Particles: Behave according to the canonical PSO update rules.
    2.  Quantum Particles: Move randomly within a "quantum cloud" (a hypersphere)
        centered around the current global best position.

    This dual mechanism balances exploitation and exploration, making the
    algorithm well-suited for dynamic optimization problems. This implementation
    is based on Section 3.2.5 and Algorithm 3.7 of Pamparà's PhD thesis.
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 swarm_size: int,
                 w: float,
                 c1: float,
                 c2: float,
                 split_ratio: float,
                 r_cloud: float,
                 **kwargs):
        """
        Initializes the Quantum Particle Swarm Optimization solver.

        Args:
            problem (Problem[List[float], float]): The dynamic optimization problem.
            name (str): the name of the solver
            swarm_size (int): The total number of particles in the swarm.
            w (float): The inertia weight for neutral particles.
            c1 (float): The cognitive coefficient for neutral particles.
            c2 (float): The social coefficient for neutral particles.
            split_ratio (float): The proportion of the swarm designated as
                neutral particles. The rest will be quantum particles.
            r_cloud (float): The radius of the quantum cloud for quantum particles.
            **kwargs: Additional keyword arguments.
        """
        # Initialize the base PSO class which sets up a full swarm
        super().__init__(problem, name, swarm_size, w, c1, c2, **kwargs)

        self.split_ratio = split_ratio
        self.r_cloud = r_cloud

        # --- Split the swarm into neutral and quantum subgroups ---
        num_neutral = int(self.swarm_size * self.split_ratio)
        
        self.neutral_indices = list(range(num_neutral))
        self.quantum_indices = list(range(num_neutral, self.swarm_size))
    
    def step(self) -> None:
        """Performs one iteration of the QPSO algorithm."""
        lower_bounds, upper_bounds = self.problem.bounds

        # --- 1. Update Neutral Particles ---
        for i in self.neutral_indices:
            # Update velocity using standard PSO equation
            for d in range(self.problem.dimension):
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i][d] - self.positions[i][d])
                social = self.c2 * r2 * (self.gbest_position[d] - self.positions[i][d])
                self.velocities[i][d] = (self.w * self.velocities[i][d]) + cognitive + social
            
            # Update position
            for d in range(self.problem.dimension):
                self.positions[i][d] += self.velocities[i][d]
                self.positions[i][d] = max(lower_bounds[d], min(self.positions[i][d], upper_bounds[d]))

            # Evaluate and update pbest/gbest
            self._evaluate_and_update_bests(i)

        # --- 2. Update Quantum Particles ---
        for i in self.quantum_indices:
            # Update position by sampling the quantum cloud (Equation 3.6)
            for d in range(self.problem.dimension):
                # Sample a uniform distribution centered on gbest with radius r_cloud
                self.positions[i][d] = random.uniform(
                    self.gbest_position[d] - self.r_cloud,
                    self.gbest_position[d] + self.r_cloud
                )
                # Ensure the particle stays within bounds
                self.positions[i][d] = max(lower_bounds[d], min(self.positions[i][d], upper_bounds[d]))

            # Evaluate and update pbest/gbest
            self._evaluate_and_update_bests(i)

    def _evaluate_and_update_bests(self, particle_idx: int):
        """
        Evaluates a particle and updates its personal best and the global best.
        """
        i = particle_idx
        # Evaluate new position
        self.evaluations[i] = self.problem.evaluate(self.positions[i])

        # Update Personal Best (pbest)
        if self.evaluations[i].fitness < self.pbest_evaluations[i].fitness:
            self.pbest_positions[i] = copy.deepcopy(self.positions[i])
            self.pbest_evaluations[i] = copy.deepcopy(self.evaluations[i])

            # Update Global Best (gbest)
            if self.pbest_evaluations[i].fitness < self.gbest_evaluation.fitness:
                self.gbest_position = copy.deepcopy(self.pbest_positions[i])
                self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[i])
