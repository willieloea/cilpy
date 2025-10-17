# cilpy/solver/solvers/pso.py

import random
from typing import List, Tuple, Any, Optional
from abc import ABC, abstractmethod

from ...problem import Problem
from .. import Solver
from .toplogy import Topology, GlobalTopology, RingTopology


class _BasePSO(Solver[List[float], float], ABC):
    """
    An abstract base class for Particle Swarm Optimization algorithms.

    This class provides the common infrastructure for PSO, including swarm
    initialization, pbest/gbest tracking, and handling of dynamic problems.
    Subclasses must implement the `_update_velocity` method to define the
    specific PSO variant's behavior.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        swarm_size: int = 30,
        c1: float = 2.05,
        c2: float = 2.05,
        topology: Optional[Topology] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, **kwargs)

        self.swarm_size = swarm_size
        self.c1, self.c2 = c1, c2

        if topology is None:
            self.topology = GlobalTopology(self.swarm_size)
        else:
            self.topology = topology

        # --- Initialize Swarm ---
        self.dimension = self.problem.get_dimension()
        self.lower_bounds, self.upper_bounds = self.problem.get_bounds()

        self.positions = self._initialize_positions()
        self.velocities = self._initialize_velocities()

        self.pbest_positions = [p[:] for p in self.positions]
        self.pbest_values = [self._evaluate_fitness(pos) for pos in self.pbest_positions]

        # Initialize gbest by finding the best among the initial pbest values.
        best_idx = 0
        for i in range(1, self.swarm_size):
            if self._is_better(self.pbest_values[i], self.pbest_values[best_idx]):
                best_idx = i

        # Make a copy of the position to avoid reference issues.
        self.gbest_position = self.pbest_positions[best_idx][:]
        self.gbest_value = self.pbest_values[best_idx]

        # Store whether the problem is dynamic to avoid checks every step
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()


    def _initialize_positions(self) -> List[List[float]]:
        """Initializes swarm positions within the problem's bounds."""
        return [
            [
                random.uniform(self.lower_bounds[d], self.upper_bounds[d])
                for d in range(self.dimension)
            ]
            for _ in range(self.swarm_size)
        ]

    def _initialize_velocities(self) -> List[List[float]]:
        """Initializes swarm velocities, typically based on search space size."""
        v_range = [(u - l) * 0.5 for l, u in zip(self.upper_bounds, self.lower_bounds)]
        return [
            [random.uniform(-vr, vr) for vr in v_range]
            for _ in range(self.swarm_size)
        ]

    def _clamp_position(self, position: List[float]) -> List[float]:
        """Ensures a particle's position stays within the defined bounds."""
        return [
            min(max(x, l), u)
            for x, l, u in zip(position, self.lower_bounds, self.upper_bounds)
        ]

    def _re_evaluate_bests(self) -> None:
        """Re-evaluates pbest and gbest fitness values if the problem is dynamic."""
        self.pbest_values = [self._evaluate_fitness(pos) for pos in self.pbest_positions]
        self.gbest_value = self._evaluate_fitness(self.gbest_position)

        # Find the best particle in the current swarm's memory
        current_best_p_idx = 0
        for i in range(1, self.swarm_size):
            if self._is_better(
                self.pbest_values[i], self.pbest_values[current_best_p_idx]
            ):
                current_best_p_idx = i

        # Check if any personal best is now better than the global best
        if self._is_better(
            self.pbest_values[current_best_p_idx], self.gbest_value
        ):
            self.gbest_position = self.pbest_positions[current_best_p_idx][:]
            self.gbest_value = self.pbest_values[current_best_p_idx]

    def _get_local_best(self, particle_index: int) -> Tuple[List[float], Any]:
        """Finds the best pbest position within a particle's neighbourhood."""
        neighbors = self.topology.get_neighbors(particle_index)
        
        best_neighbor_idx = neighbors[0]
        for neighbor_idx in neighbors[1:]:
            if self._is_better(self.pbest_values[neighbor_idx], self.pbest_values[best_neighbor_idx]):
                best_neighbor_idx = neighbor_idx
        
        return self.pbest_positions[best_neighbor_idx], self.pbest_values[best_neighbor_idx]

    def _calculate_total_violation(self, solution: List[float]) -> float:
        """
        Calculates the sum of all constraint violations for a solution.
        """
        total_violation = 0.0
        inequality_constraints, equality_constraints = self.problem.get_constraint_functions()
        # Inequality constraints are of the form g(x) <= 0
        for g in inequality_constraints:
            total_violation += max(0, g(solution))
        # Equality constraints are of the form h(x) = 0
        for h in equality_constraints:
            total_violation += abs(h(solution))
        return total_violation

    def _evaluate_fitness(self, solution: List[float]) -> Tuple[float, float]:
        """
        Evaluates a solution using Deb's rules, returning (violation, objective).
        """
        violation = self._calculate_total_violation(solution)
        # Assuming single-objective problem as per the original DebsRules
        objective = self.problem.get_objective_functions()[0](solution)
        return (violation, float(objective))

    def _is_better(self, fitness_a: Tuple[float, float], fitness_b: Tuple[float, float]) -> bool:
        """
        Compares two fitness tuples. True if fitness_a is strictly better.
        """
        return fitness_a < fitness_b

    def _repair(self, solution: List[float]) -> List[float]:
        """
        Placeholder for a repair mechanism. By default, does nothing.
        """
        return solution

    def step(self) -> None:
        """Performs one iteration of the PSO algorithm."""
        if self.is_dynamic or self.is_constrained_dynamic:
            self._re_evaluate_bests()

        for i in range(self.swarm_size):
            local_best_pos, _ = self._get_local_best(i)

            # 1. Update velocity
            self.velocities[i] = self._update_velocity(i, local_best_pos)

            # 2. Update position
            new_position = [
                pos + vel
                for pos, vel in zip(self.positions[i], self.velocities[i])
            ]
            self.positions[i] = self._clamp_position(new_position)
            self.positions[i] = self._repair(self.positions[i])

            # 3. Evaluate new position
            new_fitness = self._evaluate_fitness(self.positions[i])

            # 4. Update personal best (pbest)
            if self._is_better(new_fitness, self.pbest_values[i]):
                self.pbest_positions[i] = self.positions[i][:]
                self.pbest_values[i] = new_fitness

                # 5. Update overall global best (gbest)
                if self._is_better(new_fitness, self.gbest_value):
                    self.gbest_position = self.positions[i][:]
                    self.gbest_value = new_fitness

    @abstractmethod
    def _update_velocity(self, particle_index: int, social_best_position: List[float]) -> List[float]:
        """
        Abstract method to update the velocity of a single particle.

        This method defines the core difference between PSO variants.
        """
        pass

    def get_best(self) -> Tuple[List[float], float]:
        """
        Returns the best solution found so far and its raw objective value.
        """
        objective_func = self.problem.get_objective_functions()[0]
        raw_objective_value = objective_func(self.gbest_position)
        return self.gbest_position, raw_objective_value


# --- Concrete PSO Implementations ---


class BasePSO(_BasePSO):
    """
    The original PSO as described by Kennedy and Eberhart (1995).

    This implementation uses the basic velocity update rule without an
    inertia weight or explicit velocity clamping. The velocity of a particle
    is influenced by its previous velocity, its personal best position, and
    the global best position.
    """

    def _update_velocity(self, particle_index: int, social_best_position: List[float]) -> List[float]:
        new_velocity = []
        for d in range(self.dimension):
            r1, r2 = random.random(), random.random()
            cognitive = (
                self.c1 * r1 * (self.pbest_positions[particle_index][d] - self.positions[particle_index][d])
            )
            social = self.c2 * r2 * (social_best_position[d] - self.positions[particle_index][d])
            velocity = self.velocities[particle_index][d] + cognitive + social
            new_velocity.append(velocity)
        return new_velocity


class InertiaWeightPSO(_BasePSO):
    """
    PSO with an inertia weight term, as introduced by Shi and Eberhart.

    The inertia weight `w` controls the influence of the previous velocity on
    the new velocity, balancing global exploration (high `w`) and local
    exploitation (low `w`).
    """

    def __init__(
        self, problem: Problem[List[float], float], w: float = 0.729, **kwargs: Any
    ):
        self.w = w
        super().__init__(problem, **kwargs)

    def _update_velocity(self, particle_index: int, social_best_position: List[float]) -> List[float]:
        new_velocity = []
        for d in range(self.dimension):
            r1, r2 = random.random(), random.random()
            cognitive = (
                self.c1 * r1 * (self.pbest_positions[particle_index][d] - self.positions[particle_index][d])
            )
            social = self.c2 * r2 * (social_best_position[d] - self.positions[particle_index][d])
            velocity = (self.w * self.velocities[particle_index][d]) + cognitive + social
            new_velocity.append(velocity)
        return new_velocity


class VelocityClampingPSO(_BasePSO):
    """
    The original PSO with an explicit velocity clamping mechanism.

    This prevents the particle's velocity from growing excessively, which can
    cause it to overshoot good solutions or leave the search space entirely.
    The maximum velocity is typically set as a fraction of the search space range.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        vel_clamp_factor: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(problem, **kwargs)
        self.v_max = [
            (u - l) * vel_clamp_factor
            for l, u in zip(self.lower_bounds, self.upper_bounds)
        ]

    def _clamp_velocity(self, velocity: List[float]) -> List[float]:
        return [min(max(v, -vm), vm) for v, vm in zip(velocity, self.v_max)]

    def _update_velocity(self, particle_index: int, social_best_position: List[float]) -> List[float]:
        new_velocity = []
        for d in range(self.dimension):
            r1, r2 = random.random(), random.random()
            cognitive = (
                self.c1 * r1 * (self.pbest_positions[particle_index][d] - self.positions[particle_index][d])
            )
            social = self.c2 * r2 * (social_best_position[d] - self.positions[particle_index][d])
            velocity = self.velocities[particle_index][d] + cognitive + social
            new_velocity.append(velocity)

        return self._clamp_velocity(new_velocity)


class CanonicalPSO(_BasePSO):
    """
    The canonical PSO, incorporating both inertia weight and velocity clamping.

    This is a widely used and effective variant of PSO that combines the balancing
    effect of the inertia weight (`w`) with the stability offered by velocity
    clamping. This class replaces the original `GbestPSO`.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        w: float = 0.729,
        vel_clamp_factor: float = 0.5,
        **kwargs: Any,
    ):
        self.w = w
        super().__init__(problem, **kwargs)
        self.v_max = [
            (u - l) * vel_clamp_factor
            for l, u in zip(self.lower_bounds, self.upper_bounds)
        ]

    def _clamp_velocity(self, velocity: List[float]) -> List[float]:
        return [min(max(v, -vm), vm) for v, vm in zip(velocity, self.v_max)]

    def _update_velocity(self, particle_index: int, social_best_position: List[float]) -> List[float]:
        new_velocity = []
        for d in range(self.dimension):
            r1, r2 = random.random(), random.random()
            cognitive = (
                self.c1 * r1 * (self.pbest_positions[particle_index][d] - self.positions[particle_index][d])
            )
            social = self.c2 * r2 * (social_best_position[d] - self.positions[particle_index][d])
            velocity = (self.w * self.velocities[particle_index][d]) + cognitive + social
            new_velocity.append(velocity)

        return self._clamp_velocity(new_velocity)

# --- Lbest Implementations ---

class LbestCanonicalPSO(CanonicalPSO):
    """
    The canonical PSO with a local best (lbest) ring topology.
    """
    def __init__(self, problem: Problem, k: int = 1, **kwargs):
        # We need swarm_size to initialize the topology, so we get it from kwargs
        swarm_size = kwargs.get('swarm_size', 30)
        topology = RingTopology(swarm_size=swarm_size, k=k)
        super().__init__(problem, topology=topology, **kwargs)

class LbestInertiaWeightPSO(InertiaWeightPSO):
    """
    PSO with inertia weight and a local best (lbest) ring topology.
    """
    def __init__(self, problem: Problem, k: int = 1, **kwargs):
        swarm_size = kwargs.get('swarm_size', 30)
        topology = RingTopology(swarm_size=swarm_size, k=k)
        super().__init__(problem, topology=topology, **kwargs)
