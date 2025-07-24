import random
from typing import List, Tuple, Any

from ..problem import Problem
from . import Solver


class CCPSO(Solver[List[float]]):
    """
    Co-evolutionary Particle Swarm Optimisation (CCPSO) for constrained problems.

    This algorithm, inspired by Shi and Krohling, uses a co-evolutionary
    approach to handle constraints. It decomposes the problem into two
    sub-problems, solved by two cooperating PSO swarms:

    1.  **Objective Swarm (P1)**: Searches the solution space for variables 'x'.
        Its goal is to MINIMIZE the Lagrangian function.
    2.  **Penalty Swarm (P2)**: Searches the space of Lagrangian multipliers
        ('μ' for inequality, 'λ' for equality constraints). Its goal is to
        MAXIMIZE the Lagrangian function.

    The fitness of each particle in one swarm is evaluated using the global best
    solution from the other swarm. This creates a co-evolutionary, min-max
    dynamic.

    This implementation is also adapted for dynamic constrained optimization
    problems (DCOPs) by re-evaluating personal and global bests if the
    problem's landscape changes.

    References:
        Y. Shi and R. A. Krohling. (2002). “Co-Evolutionary Particle Swarm
        Optimization To Solve Min-Max Problems”.
    """

    def __init__(
        self,
        problem: Problem[List[float]],
        swarm_size_x: int = 30,
        swarm_size_l: int = 30,
        w: float = 0.729,
        c1: float = 2.05,
        c2: float = 2.05,
        lambda_bounds: Tuple[float, float] = (0.0, 1000.0),
        **kwargs: Any
    ):
        """
        Initializes the CCPSO solver.

        Args:
            problem (Problem[List[float]]): The constrained optimization problem.
            swarm_size_x (int): The number of particles in the objective swarm.
            swarm_size_l (int): The number of particles in the penalty swarm.
            w (float): Inertia weight for both PSOs.
            c1 (float): Cognitive coefficient for both PSOs.
            c2 (float): Social coefficient for both PSOs.
            lambda_bounds (Tuple[float, float]): The search space for Lagrangian
                                                 multipliers.
            **kwargs: Additional arguments for the base Solver class.
        """
        super().__init__(problem, **kwargs)

        # PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm_size_x = swarm_size_x
        self.swarm_size_l = swarm_size_l

        # Problem-specific information
        self.objective = self.problem.get_objective_functions()[0]
        self.inequality_constraints, self.equality_constraints = (
            self.problem.get_constraint_functions()
        )
        self.lambda_dim = len(self.inequality_constraints) + len(
            self.equality_constraints
        )
        self.lambda_bounds = lambda_bounds
        self.is_dynamic, self.is_constrained_dynamic = self.problem.is_dynamic()

        # --- Initialize Objective Swarm (P1 for x) ---
        self.positions_x = [
            self.problem.initialize_solution() for _ in range(self.swarm_size_x)
        ]
        self.velocities_x = [
            self._initialize_velocity(self.problem.get_bounds())
            for _ in range(self.swarm_size_x)
        ]
        self.pbest_positions_x = [p.copy() for p in self.positions_x]

        # --- Initialize Penalty Swarm (P2 for μ, λ) ---
        if self.lambda_dim > 0:
            self.positions_l = [
                self._initialize_lambda_particle() for _ in range(self.swarm_size_l)
            ]
            l_lower_bounds = [self.lambda_bounds[0]] * self.lambda_dim
            l_upper_bounds = [self.lambda_bounds[1]] * self.lambda_dim
            self.bounds_l = (l_lower_bounds, l_upper_bounds)  # <-- Note the 'self.'
            self.velocities_l = [
                self._initialize_velocity(self.bounds_l)
                for _ in range(self.swarm_size_l)
            ]  # <-- Use self.bounds_l
            self.pbest_positions_l = [p.copy() for p in self.positions_l]
            # Initialize a placeholder gbest lambda for the first evaluation of swarm X
            self.gbest_position_l = self._initialize_lambda_particle()
        else:  # Unconstrained problem
            self.positions_l, self.velocities_l, self.pbest_positions_l = [], [], []
            self.gbest_position_l = []
            self.bounds_l = ([], [])  # Initialize to avoid attribute errors

        # --- Initial Evaluation and Best Finding ---
        # Evaluate pbest_x using the initial placeholder gbest_l
        self.pbest_values_x = [
            self._calculate_lagrangian(pos, self.gbest_position_l)
            for pos in self.pbest_positions_x
        ]

        # Find gbest_x based on initial evaluation
        gbest_idx_x = min(
            range(self.swarm_size_x), key=lambda i: self.pbest_values_x[i]
        )
        self.gbest_position_x = self.pbest_positions_x[gbest_idx_x]
        self.gbest_value_x = self.pbest_values_x[gbest_idx_x]

        if self.lambda_dim > 0:
            # Evaluate pbest_l using the newly found gbest_x
            self.pbest_values_l = [
                self._calculate_lagrangian(self.gbest_position_x, pos)
                for pos in self.pbest_positions_l
            ]

            # Find gbest_l (maximization)
            gbest_idx_l = max(
                range(self.swarm_size_l), key=lambda i: self.pbest_values_l[i]
            )
            self.gbest_position_l = self.pbest_positions_l[gbest_idx_l]
            self.gbest_value_l = self.pbest_values_l[gbest_idx_l]

    def _calculate_lagrangian(self, x: List[float], lag_mult: List[float]) -> float:
        """
        Calculates the value of the Lagrangian function L(x, μ, λ).
        L(x, μ, λ) = f(x) + Σ(μ_i * max(0, g_i(x))) + Σ(λ_j * |h_j(x)|)
        """
        obj_val = self.objective(x)

        if not self.lambda_dim:
            return obj_val

        # Unpack multipliers
        num_ineq = len(self.inequality_constraints)
        mu = lag_mult[:num_ineq]
        lam = lag_mult[num_ineq:]

        # Inequality constraint penalty (g_i(x) <= 0)
        ineq_penalty = sum(
            m * max(0, const(x)) for m, const in zip(mu, self.inequality_constraints)
        )

        # Equality constraint penalty (h_j(x) = 0)
        eq_penalty = sum(
            l * abs(const(x)) for l, const in zip(lam, self.equality_constraints)
        )

        return obj_val + ineq_penalty + eq_penalty

    def _initialize_velocity(
        self, bounds: Tuple[List[float], List[float]]
    ) -> List[float]:
        lower_bounds, upper_bounds = bounds  # Unpack the tuple of lists
        max_velocity = [abs(u - l) * 0.5 for l, u in zip(lower_bounds, upper_bounds)]
        return [random.uniform(-v, v) for v in max_velocity]

    def _initialize_lambda_particle(self) -> List[float]:
        return [random.uniform(*self.lambda_bounds) for _ in range(self.lambda_dim)]

    def _clamp_position(
        self, position: List[float], bounds: Tuple[List[float], List[float]]
    ) -> List[float]:
        lower_bounds, upper_bounds = bounds
        return [
            min(max(x, l), u) for x, l, u in zip(position, lower_bounds, upper_bounds)
        ]

    # def _clamp_position(self, position: List[float], bounds: Any) -> List[float]:
    #     return [min(max(x, b[0]), b[1]) for x, b in zip(position, bounds)]

    def step(self) -> None:
        """Performs one co-evolutionary iteration."""
        if self.is_dynamic or self.is_constrained_dynamic:
            # Re-evaluate all personal bests and global bests as fitness may have changed
            self.pbest_values_x = [
                self._calculate_lagrangian(pos, self.gbest_position_l)
                for pos in self.pbest_positions_x
            ]
            self.gbest_value_x = self._calculate_lagrangian(
                self.gbest_position_x, self.gbest_position_l
            )

            # Check if any other pbest_x is now better than the old gbest_x
            current_best_idx_x = min(
                range(self.swarm_size_x), key=lambda i: self.pbest_values_x[i]
            )
            if self.pbest_values_x[current_best_idx_x] < self.gbest_value_x:
                self.gbest_position_x = self.pbest_positions_x[current_best_idx_x]
                self.gbest_value_x = self.pbest_values_x[current_best_idx_x]

            if self.lambda_dim > 0:
                self.pbest_values_l = [
                    self._calculate_lagrangian(self.gbest_position_x, pos)
                    for pos in self.pbest_positions_l
                ]
                self.gbest_value_l = self._calculate_lagrangian(
                    self.gbest_position_x, self.gbest_position_l
                )

                # Check if any other pbest_l is now better than the old gbest_l (maximization)
                current_best_idx_l = max(
                    range(self.swarm_size_l), key=lambda i: self.pbest_values_l[i]
                )
                if self.pbest_values_l[current_best_idx_l] > self.gbest_value_l:
                    self.gbest_position_l = self.pbest_positions_l[current_best_idx_l]
                    self.gbest_value_l = self.pbest_values_l[current_best_idx_l]

        # --- 1. Update Objective Swarm (P1) - Minimization ---
        dim_x = self.problem.get_dimension()
        bounds_x = self.problem.get_bounds()
        for i in range(self.swarm_size_x):
            # Update velocity
            new_velocity = []
            for d in range(dim_x):
                r1, r2 = random.random(), random.random()
                cognitive = (
                    self.c1
                    * r1
                    * (self.pbest_positions_x[i][d] - self.positions_x[i][d])
                )
                social = (
                    self.c2 * r2 * (self.gbest_position_x[d] - self.positions_x[i][d])
                )
                velocity = self.w * self.velocities_x[i][d] + cognitive + social
                new_velocity.append(velocity)
            self.velocities_x[i] = new_velocity

            # Update position
            new_position = [
                pos + vel for pos, vel in zip(self.positions_x[i], self.velocities_x[i])
            ]
            self.positions_x[i] = self._clamp_position(new_position, bounds_x)

            # Evaluate using gbest from penalty swarm
            new_fitness = self._calculate_lagrangian(
                self.positions_x[i], self.gbest_position_l
            )

            # Update personal best (minimization)
            if new_fitness < self.pbest_values_x[i]:
                self.pbest_positions_x[i] = self.positions_x[i]
                self.pbest_values_x[i] = new_fitness

                # Update global best
                if new_fitness < self.gbest_value_x:
                    self.gbest_position_x = self.positions_x[i]
                    self.gbest_value_x = new_fitness

        # --- 2. Update Penalty Swarm (P2) - Maximization ---
        if self.lambda_dim > 0:
            for i in range(self.swarm_size_l):
                # Update velocity
                new_velocity = []
                for d in range(self.lambda_dim):
                    r1, r2 = random.random(), random.random()
                    cognitive = (
                        self.c1
                        * r1
                        * (self.pbest_positions_l[i][d] - self.positions_l[i][d])
                    )
                    social = (
                        self.c2
                        * r2
                        * (self.gbest_position_l[d] - self.positions_l[i][d])
                    )
                    velocity = self.w * self.velocities_l[i][d] + cognitive + social
                    new_velocity.append(velocity)
                self.velocities_l[i] = new_velocity

                # Update position
                new_position = [
                    pos + vel
                    for pos, vel in zip(self.positions_l[i], self.velocities_l[i])
                ]
                self.positions_l[i] = self._clamp_position(new_position, self.bounds_l)

                # Evaluate using gbest from objective swarm
                new_fitness = self._calculate_lagrangian(
                    self.gbest_position_x, self.positions_l[i]
                )

                # Update personal best (maximization)
                if new_fitness > self.pbest_values_l[i]:
                    self.pbest_positions_l[i] = self.positions_l[i]
                    self.pbest_values_l[i] = new_fitness

                    # Update global best
                    if new_fitness > self.gbest_value_l:
                        self.gbest_position_l = self.positions_l[i]
                        self.gbest_value_l = new_fitness

    def get_best(self) -> Tuple[List[float], List[float]]:
        """
        Returns the best solution 'x' and its raw objective value (not the
        Lagrangian value).
        """
        # The user is interested in the objective function value of the best
        # solution, not its Lagrangian fitness.
        best_objective_value = self.objective(self.gbest_position_x)
        return self.gbest_position_x, [best_objective_value]
