# cilpy/solver/solvers/ccpso.py

import random
import numpy as np
from typing import List, Tuple, Any

from ...problem import Problem
from .. import Solver


class CCPSO(Solver[np.ndarray, float]):
    """
    Co-evolutionary Particle Swarm Optimisation (CCPSO) for constrained problems.

    This algorithm, inspired by Shi and Krohling, uses a co-evolutionary
    approach to handle constraints. It decomposes the problem into two
    sub-problems, solved by two cooperating PSO swarms:

    1.  **Objective Swarm (P1)**: Searches the solution space 'x'. Its goal
        is to MINIMIZE the Lagrangian function.
    2.  **Penalty Swarm (P2)**: Searches the space of Lagrangian multipliers
        ('μ' for inequality, 'λ' for equality). Its goal is to MAXIMIZE the
        Lagrangian function.

    The fitness of each particle in one swarm is evaluated using the global best
    solution from the other swarm, creating a min-max dynamic. This
    implementation is also adapted for dynamic constrained optimization
    problems (DCOPs) by re-evaluating personal and global bests when the
    environment changes.

    References:
        Y. Shi and R. A. Krohling. (2002). “Co-Evolutionary Particle Swarm
        Optimization To Solve Min-Max Problems”.
    """

    def __init__(
        self,
        problem: Problem[np.ndarray, float],
        swarm_size_x: int = 30,
        swarm_size_l: int = 30,
        w: float = 0.7298,
        c1: float = 1.49618,
        c2: float = 1.49618,
        lambda_bounds: Tuple[float, float] = (0.0, 1000.0),
        **kwargs: Any
    ):
        super().__init__(problem, **kwargs)
        self.w, self.c1, self.c2 = w, c1, c2
        self.swarm_size_x = swarm_size_x
        self.swarm_size_l = swarm_size_l

        self.objective = self.problem.get_objective_functions()[0]
        self.ineq_constraints, self.eq_constraints = self.problem.get_constraint_functions()
        self.lambda_dim = len(self.ineq_constraints) + len(self.eq_constraints)
        self.lambda_bounds = lambda_bounds
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

        # --- Initialize Objective Swarm (P1 for x) ---
        dim_x = self.problem.get_dimension()
        bounds_x = self.problem.get_bounds()
        self.positions_x = np.array([
            np.random.uniform(bounds_x[0], bounds_x[1], dim_x)
            for _ in range(self.swarm_size_x)
        ])
        self.velocities_x = np.array([self._initialize_velocity(bounds_x) for _ in range(self.swarm_size_x)])
        self.pbest_positions_x = self.positions_x.copy()

        # --- Initialize Penalty Swarm (P2 for μ, λ) ---
        if self.lambda_dim > 0:
            self.bounds_l = (
                np.full(self.lambda_dim, self.lambda_bounds[0]),
                np.full(self.lambda_dim, self.lambda_bounds[1])
            )
            self.positions_l = np.array([
                np.random.uniform(self.bounds_l[0], self.bounds_l[1], self.lambda_dim)
                for _ in range(self.swarm_size_l)
            ])
            self.velocities_l = np.array([self._initialize_velocity(self.bounds_l) for _ in range(self.swarm_size_l)])
            self.pbest_positions_l = self.positions_l.copy()
            self.gbest_position_l = self.pbest_positions_l[0].copy()
        else:
            self.bounds_l = (np.array([]), np.array([]))
            self.positions_l, self.velocities_l, self.pbest_positions_l = np.array([]), np.array([]), np.array([])
            self.gbest_position_l = np.array([])

        # --- Initial Evaluation and Best Finding ---
        # 1. Evaluate pbest_x using a placeholder gbest_l
        self.pbest_values_x = np.array([
            self._calculate_lagrangian(pos, self.gbest_position_l)
            for pos in self.pbest_positions_x
        ])
        
        # 2. Find the initial gbest_x
        gbest_idx_x = np.argmin(self.pbest_values_x)
        self.gbest_position_x = self.pbest_positions_x[gbest_idx_x].copy()
        self.gbest_value_x = self.pbest_values_x[gbest_idx_x]

        if self.lambda_dim > 0:
            # 3. Evaluate pbest_l using the real gbest_x
            self.pbest_values_l = np.array([
                self._calculate_lagrangian(self.gbest_position_x, pos)
                for pos in self.pbest_positions_l
            ])
            
            # 4. Find the initial gbest_l (maximization)
            gbest_idx_l = np.argmax(self.pbest_values_l)
            self.gbest_position_l = self.pbest_positions_l[gbest_idx_l].copy()
            self.gbest_value_l = self.pbest_values_l[gbest_idx_l]

    def _calculate_lagrangian(self, x: np.ndarray, lag_mult: np.ndarray) -> float:
        obj_val = self.objective(x)
        if self.lambda_dim == 0:
            return obj_val

        num_ineq = len(self.ineq_constraints)
        mu = lag_mult[:num_ineq]
        lam = lag_mult[num_ineq:]

        ineq_penalty = sum(m * max(0, const(x)) for m, const in zip(mu, self.ineq_constraints))
        eq_penalty = sum(l * abs(const(x)) for l, const in zip(lam, self.eq_constraints))

        return obj_val + ineq_penalty + eq_penalty

    def _initialize_velocity(self, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        lower, upper = bounds
        max_velocity = (upper - lower) * 0.5
        return np.random.uniform(-max_velocity, max_velocity, len(lower))

    def _clamp_position(self, position: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return np.clip(position, bounds[0], bounds[1])

    def step(self) -> None:
        if self.is_dynamic or self.is_constrained_dynamic:
            self._reevaluate_bests()

        # --- 1. Update Objective Swarm (P1) - Minimization ---
        self._update_swarm_x()

        # --- 2. Update Penalty Swarm (P2) - Maximization ---
        if self.lambda_dim > 0:
            self._update_swarm_l()

    def _reevaluate_bests(self):
        """Re-evaluates all personal and global bests."""
        self.pbest_values_x = np.array([self._calculate_lagrangian(pos, self.gbest_position_l) for pos in self.pbest_positions_x])
        self.gbest_value_x = self._calculate_lagrangian(self.gbest_position_x, self.gbest_position_l)
        
        current_best_idx_x = np.argmin(self.pbest_values_x)
        if self.pbest_values_x[current_best_idx_x] < self.gbest_value_x:
            self.gbest_position_x = self.pbest_positions_x[current_best_idx_x].copy()
            self.gbest_value_x = self.pbest_values_x[current_best_idx_x]

        if self.lambda_dim > 0:
            self.pbest_values_l = np.array([self._calculate_lagrangian(self.gbest_position_x, pos) for pos in self.pbest_positions_l])
            self.gbest_value_l = self._calculate_lagrangian(self.gbest_position_x, self.gbest_position_l)
            
            current_best_idx_l = np.argmax(self.pbest_values_l)
            if self.pbest_values_l[current_best_idx_l] > self.gbest_value_l:
                self.gbest_position_l = self.pbest_positions_l[current_best_idx_l].copy()
                self.gbest_value_l = self.pbest_values_l[current_best_idx_l]

    def _update_swarm_x(self):
        r1, r2 = np.random.rand(self.swarm_size_x, 1), np.random.rand(self.swarm_size_x, 1)
        cognitive = self.c1 * r1 * (self.pbest_positions_x - self.positions_x)
        social = self.c2 * r2 * (self.gbest_position_x - self.positions_x)
        self.velocities_x = self.w * self.velocities_x + cognitive + social
        self.positions_x += self.velocities_x
        self.positions_x = self._clamp_position(self.positions_x, self.problem.get_bounds())

        new_fitness_values = np.array([self._calculate_lagrangian(p, self.gbest_position_l) for p in self.positions_x])
        
        # Update personal bests
        improvement_mask = new_fitness_values < self.pbest_values_x
        self.pbest_positions_x[improvement_mask] = self.positions_x[improvement_mask]
        self.pbest_values_x[improvement_mask] = new_fitness_values[improvement_mask]
        
        # Update global best
        min_idx = np.argmin(self.pbest_values_x)
        if self.pbest_values_x[min_idx] < self.gbest_value_x:
            self.gbest_value_x = self.pbest_values_x[min_idx]
            self.gbest_position_x = self.pbest_positions_x[min_idx].copy()
            
    def _update_swarm_l(self):
        r1, r2 = np.random.rand(self.swarm_size_l, 1), np.random.rand(self.swarm_size_l, 1)
        cognitive = self.c1 * r1 * (self.pbest_positions_l - self.positions_l)
        social = self.c2 * r2 * (self.gbest_position_l - self.positions_l)
        self.velocities_l = self.w * self.velocities_l + cognitive + social
        self.positions_l += self.velocities_l
        self.positions_l = self._clamp_position(self.positions_l, self.bounds_l)
        
        new_fitness_values = np.array([self._calculate_lagrangian(self.gbest_position_x, p) for p in self.positions_l])

        # Update personal bests (maximization)
        improvement_mask = new_fitness_values > self.pbest_values_l
        self.pbest_positions_l[improvement_mask] = self.positions_l[improvement_mask]
        self.pbest_values_l[improvement_mask] = new_fitness_values[improvement_mask]

        # Update global best (maximization)
        max_idx = np.argmax(self.pbest_values_l)
        if self.pbest_values_l[max_idx] > self.gbest_value_l:
            self.gbest_value_l = self.pbest_values_l[max_idx]
            self.gbest_position_l = self.pbest_positions_l[max_idx].copy()

    def get_best(self) -> Tuple[np.ndarray, float]:
        """Returns the best solution 'x' and its raw objective value."""
        best_objective_value = self.objective(self.gbest_position_x)
        return self.gbest_position_x, float(best_objective_value)
