# cilpy/problem/cmpb.py

import numpy as np
from typing import Any, Callable, Dict, List, Tuple

from . import Problem
from .mpb import MovingPeaksBenchmark


class ConstrainedMovingPeaksBenchmark(Problem[np.ndarray, np.float64]):
    """
    An implementation of the Constrained Moving Peaks Benchmark (CMPB).

    This class generates a dynamic constrained optimization problem by composing
    two independent Moving Peaks Benchmark instances: one for the objective
    function landscape (f) and one for the constraint landscape (g).

    The final optimization problem is derived from maximizing h(x) = f(x) - g(x),
    where a solution is feasible if h(x) >= 0 (i.e., f(x) >= g(x)).

    For standard minimization solvers, the problem is formulated as:
    Minimize: g(x) - f(x)
    Subject to: g(x) - f(x) <= 0

    This implementation relies on the composition of two `MovingPeaksBenchmark`
    instances. The dynamic nature of the problem is handled by invoking the
    fitness functions of the underlying landscapes, which in turn manage their
    internal state and evaluation counters.

    NOTE: This implementation assumes that for any given solution, the objective
    function is evaluated before the constraint function(s). This ensures that
    the environment state is updated exactly once per evaluation cycle.

    References:
        Chapter 5, Constrained Moving Peaks Benchmark, from the provided document.
    """

    def __init__(
        self,
        f_params: Dict[str, Any],
        g_params: Dict[str, Any],
        name: str = "ConstrainedMovingPeaksBenchmark",
    ):
        """
        Initializes the Constrained Moving Peaks Benchmark generator.

        Args:
            f_params (Dict[str, Any]): A dictionary of parameters for the
                objective landscape (f), passed to the MovingPeaksBenchmark
                constructor.
            g_params (Dict[str, Any]): A dictionary of parameters for the
                constraint landscape (g), passed to the MovingPeaksBenchmark
                constructor.
            name (str): A name for the problem instance.
        """
        f_dim = f_params.get("dimension")
        g_dim = g_params.get("dimension")

        if f_dim is None or g_dim is None or f_dim != g_dim:
            raise ValueError(
                "The 'dimension' parameter must be specified and identical for "
                "both f_params and g_params."
            )

        self.f_landscape = MovingPeaksBenchmark(**f_params)
        self.g_landscape = MovingPeaksBenchmark(**g_params)

        # The problem definition is based on the domain of the 'f' landscape.
        super().__init__(
            dimension=self.f_landscape.get_dimension(),
            bounds=self.f_landscape.get_bounds(),
            name=name,
        )

        # Determine if landscapes are dynamic based on their change frequency
        self._is_objective_dynamic = f_params.get("change_frequency", 0) > 0
        self._is_constraint_dynamic = g_params.get("change_frequency", 0) > 0

    def _objective(self, x: np.ndarray) -> np.float64:
        """
        The composed objective function for minimization: g(x) - f(x).

        This function is responsible for triggering the environment change in
        the underlying landscapes by calling their respective `_fitness` methods.
        """
        # Calling _fitness increments the internal evaluation counter and may
        # trigger an environment update. We get the negated maximization value.
        f_neg_val = self.f_landscape._fitness(x)
        g_neg_val = self.g_landscape._fitness(x)

        # The objective for minimization is g(x) - f(x).
        # This is equivalent to (-f(x)) - (-g(x)).
        return f_neg_val - g_neg_val

    def _constraint(self, x: np.ndarray) -> np.float64:
        """
        The inequality constraint function: g(x) - f(x) <= 0.

        This function uses the raw value getters to avoid triggering a second
        environment update, assuming _objective() was called first.
        """
        # Use the raw maximization value to avoid incrementing the eval counter
        # a second time for the same solution.
        f_val = self.f_landscape._get_raw_maximization_value(x)
        g_val = self.g_landscape._get_raw_maximization_value(x)

        # The constraint is feasible if g(x) - f(x) <= 0.
        return g_val - f_val

    def get_objective_functions(self) -> List[Callable[[np.ndarray], np.float64]]:
        """Returns the objective function for minimization."""
        return [self._objective]

    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        """Returns the inequality constraint `g(x) - f(x) <= 0`."""
        # One inequality constraint, no equality constraints.
        return ([self._constraint], [])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the search space boundaries from the 'f' landscape."""
        return self.bounds

    def get_dimension(self) -> int:
        """Returns the dimensionality of the problem."""
        return self.dimension

    def is_dynamic(self) -> Tuple[bool, bool]:
        """
        Indicates whether the problem has dynamic objectives or constraints.
        The objective is considered dynamic if the 'f' landscape can change.
        The constraint is considered dynamic if either 'f' or 'g' can change.
        """
        is_constraint_dynamic = self._is_objective_dynamic or self._is_constraint_dynamic
        return (self._is_objective_dynamic, is_constraint_dynamic)