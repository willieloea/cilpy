# cilpy/solver/solvers/pso.py

import random
from typing import List, Tuple, Any, Optional
from abc import ABC, abstractmethod

from ...problem import Problem, Evaluation
from .. import Solver
from .toplogy import Topology, GlobalTopology, RingTopology


class StandardComparisonMixin:
    """
    A mixin for unconstrained problems. Compares solutions based on fitness only.
    """
    def _is_better(self, eval1: Evaluation[float], eval2: Evaluation[float]) -> bool:
        """Compares solutions based on fitness only (assumes minimization)."""
        return eval1.fitness < eval2.fitness


class DebsRulesComparisonMixin:
    """
    A mixin that provides a comparison method based on Deb's feasibility rules.
    """
    def _calculate_total_violation(self, evaluation: Evaluation[float]) -> float:
        """Helper to calculate total violation from an Evaluation object."""
        if evaluation.constraints_inequality is None and evaluation.constraints_equality is None:
            return 0.0
        
        total_violation = 0.0
        if evaluation.constraints_inequality:
            for g_violation in evaluation.constraints_inequality:
                total_violation += max(0, g_violation)
        if evaluation.constraints_equality:
            for h_violation in evaluation.constraints_equality:
                total_violation += abs(h_violation)
        return total_violation

    def _is_better(self, eval1: Evaluation[float], eval2: Evaluation[float]) -> bool:
        """Compares two evaluations using Deb's rules."""
        violation1 = self._calculate_total_violation(eval1)
        violation2 = self._calculate_total_violation(eval2)

        is_1_feasible = (violation1 == 0)
        is_2_feasible = (violation2 == 0)

        # Rule 1: Feasible is better than infeasible.
        if is_1_feasible and not is_2_feasible:
            return True
        if not is_1_feasible and is_2_feasible:
            return False

        # Rule 2: For two infeasible, less violation is better.
        if not is_1_feasible and not is_2_feasible:
            return violation1 < violation2

        # Rule 3: For two feasible, better fitness is better.
        return eval1.fitness < eval2.fitness


class _BasePSO(Solver[List[float], float], ABC):
    """
    An abstract base class for Particle Swarm Optimization algorithms.

    This class provides the common infrastructure for PSO, including swarm
    initialization, pbest/gbest tracking, and handling of dynamic problems.
    Subclasses must implement:
    - `_update_velocity`: To define the specific PSO variant's movement logic.
    - `_is_better`: To define how two solutions are compared.
    """


    def __init__(
        self,
        problem: Problem[List[float], float],
        swarm_size: int = 30,
        c1: float = 2.05,
        c2: float = 2.05,
        topology: Optional[Any] = None, # Using Any for Topology for now
        **kwargs: Any,
    ):
        super().__init__(problem, **kwargs)

        self.swarm_size = swarm_size
        self.c1, self.c2 = c1, c2
        self.topology = topology if topology is not None else GlobalTopology(self.swarm_size)

        # --- Initialize Swarm ---
        self.dimension = self.problem.dimension
        self.lower_bounds, self.upper_bounds = self.problem.bounds

        self.positions = self._initialize_positions() # Assumes this method exists
        self.velocities = self._initialize_velocities() # Assumes this method exists

        # Store the full evaluation object for each particle's personal best
        self.pbest_positions = [p[:] for p in self.positions]
        self.pbest_evals: List[Evaluation[float]] = [self.problem.evaluate(pos) for pos in self.pbest_positions]

        # Initialize gbest by finding the best among the initial pbest evaluations.
        best_idx = 0
        for i in range(1, self.swarm_size):
            if self._is_better(self.pbest_evals[i], self.pbest_evals[best_idx]):
                best_idx = i

        self.gbest_position = self.pbest_positions[best_idx][:]
        self.gbest_eval = self.pbest_evals[best_idx]

        self.is_dynamic, _ = self.problem.is_dynamic()


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
        """Re-evaluates pbest and gbest evaluations if the problem is dynamic."""
        self.pbest_evals = [self.problem.evaluate(pos) for pos in self.pbest_positions]
        self.gbest_eval = self.problem.evaluate(self.gbest_position)

        current_best_p_idx = 0
        for i in range(1, self.swarm_size):
            if self._is_better(self.pbest_evals[i], self.pbest_evals[current_best_p_idx]):
                current_best_p_idx = i

        if self._is_better(self.pbest_evals[current_best_p_idx], self.gbest_eval):
            self.gbest_position = self.pbest_positions[current_best_p_idx][:]
            self.gbest_eval = self.pbest_evals[current_best_p_idx]

    def _get_local_best(self, particle_index: int) -> Tuple[List[float], Evaluation[float]]:
        """Finds the best pbest position within a particle's neighborhood."""
        neighbors = self.topology.get_neighbors(particle_index)
        
        best_neighbor_idx = neighbors[0]
        for neighbor_idx in neighbors[1:]:
            if self._is_better(self.pbest_evals[neighbor_idx], self.pbest_evals[best_neighbor_idx]):
                best_neighbor_idx = neighbor_idx
        
        return self.pbest_positions[best_neighbor_idx], self.pbest_evals[best_neighbor_idx]

    @abstractmethod
    def _is_better(self, eval1: Evaluation[float], eval2: Evaluation[float]) -> bool:
        """
        Abstract method for comparing two solution evaluations.

        Returns True if eval1 is better than eval2.
        """
        pass

    def _repair(self, solution: List[float]) -> List[float]:
        """
        Placeholder for a repair mechanism. By default, does nothing.
        """
        return solution

    def step(self) -> None:
        """
        Performs one full iteration of the PSO algorithm.
        """
        if self.is_dynamic:
            self._re_evaluate_bests()

        # Find the overall best particle to update gbest *after* the loop
        best_pbest_idx_in_swarm = 0

        for i in range(self.swarm_size):
            # 1. Get the social best for this particle based on topology
            lbest_position, _ = self._get_local_best(i)

            # 2. Update velocity using the concrete implementation
            self.velocities[i] = self._update_velocity(i, lbest_position)

            # 3. Update position
            for d in range(self.dimension):
                self.positions[i][d] += self.velocities[i][d]
            
            # Simple boundary enforcement
            self.positions[i] = [min(max(p, self.lower_bounds[d]), self.upper_bounds[d]) for d, p in enumerate(self.positions[i])]

            # 4. Evaluate the new position
            new_eval = self.problem.evaluate(self.positions[i])

            # 5. Update personal best (pbest) using the comparison strategy
            if self._is_better(new_eval, self.pbest_evals[i]):
                self.pbest_positions[i] = self.positions[i][:]
                self.pbest_evals[i] = new_eval
            
            # Keep track of the best pbest in this generation
            if self._is_better(self.pbest_evals[i], self.pbest_evals[best_pbest_idx_in_swarm]):
                best_pbest_idx_in_swarm = i
        
        # 6. Update global best (gbest)
        if self._is_better(self.pbest_evals[best_pbest_idx_in_swarm], self.gbest_eval):
            self.gbest_position = self.pbest_positions[best_pbest_idx_in_swarm][:]
            self.gbest_eval = self.pbest_evals[best_pbest_idx_in_swarm]


    @abstractmethod
    def _update_velocity(self, particle_index: int, social_best_position: List[float]) -> List[float]:
        """
        Abstract method to update the velocity of a single particle.

        This method defines the core difference between PSO variants.
        """
        pass

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """
        Returns the best solution found so far and its full evaluation.
        Conforms to the new Solver interface. For a single-objective PSO,
        this is a list containing one tuple.
        """
        return [(self.gbest_position, self.gbest_eval)]



# --- Concrete PSO Implementations ---

class BasePSOStrategy(_BasePSO):
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

class InertiaWeightPSOStrategy(_BasePSO):
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

class CanonicalPSOStrategy(_BasePSO):
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

# --- Now, Create COMPLETE Solvers by Combining a Strategy and a Mixin ---

# A. For UNCONSTRAINED Problems
class StandardCanonicalPSO(StandardComparisonMixin, CanonicalPSOStrategy):
    """
    The canonical PSO for unconstrained problems.
    Combines the Canonical velocity rule with standard fitness comparison.
    """
    def __init__(self, problem: Problem, **kwargs):
        # The 'c1', 'c2', 'w', etc. args are passed through kwargs to the strategy
        super().__init__(problem=problem, **kwargs)

# B. For CONSTRAINED Problems
class ConstrainedCanonicalPSO(DebsRulesComparisonMixin, CanonicalPSOStrategy):
    """
    The canonical PSO for constrained problems.
    Combines the Canonical velocity rule with Deb's rules for comparison.
    """
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem=problem, **kwargs)


class StandardInertiaWeightPSO(StandardComparisonMixin, InertiaWeightPSOStrategy):
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem=problem, **kwargs)

class ConstrainedInertiaWeightPSO(DebsRulesComparisonMixin, InertiaWeightPSOStrategy):
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem=problem, **kwargs)

# --- How to Update Your Lbest Implementations ---

class LbestCanonicalPSOStrategy(CanonicalPSOStrategy):
    """
    The canonical PSO velocity strategy configured with a local best ring topology.
    Still abstract because it lacks a comparison method.
    """
    def __init__(self, problem: Problem, k: int = 1, **kwargs):
        swarm_size = kwargs.get('swarm_size', 30)
        topology = RingTopology(swarm_size=swarm_size, k=k)
        super().__init__(problem, topology=topology, **kwargs)


# Now create a complete, usable solver from it:
class ConstrainedLbestCanonicalPSO(DebsRulesComparisonMixin, LbestCanonicalPSOStrategy):
    """
    A complete Lbest PSO for constrained problems.
    """
    def __init__(self, problem: Problem, **kwargs):
        super().__init__(problem=problem, **kwargs)
