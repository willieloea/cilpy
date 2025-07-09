# cilpy/problem/cmpb.py

from typing import Any, Callable, Dict, List, Tuple

from . import Problem
from .mpb import MovingPeaksBenchmark

class ConstrainedMovingPeaksBenchmark(Problem[List[float]]):
    """
    An implementation of the Constrained Moving Peaks Benchmark (CMPB).

    This class generates a dynamic constrained optimization problem by composing
    two independent Moving Peaks Benchmark instances: one for the objective
    function (f) and one for the constraint landscape (g).

    The final optimization problem is to maximize h(x) = f(x) - g(x),
    subject to the constraint h(x) >= 0 (i.e., f(x) >= g(x)).

    For standard minimization solvers, the problem is formulated as:
    Minimize -h(x) = g(x) - f(x)
    Subject to the inequality constraint: g(x) - f(x) <= 0.

    Reference:
    Pamparà, P. (2021). "Dynamic Co-Evolutionary Algorithms for Dynamic,
    Constrained Optimisation Problems". Chapter 5.
    """

    def __init__(self,
                 f_params: Dict[str, Any],
                 g_params: Dict[str, Any],
                 problem_name: str = "ConstrainedMovingPeaksBenchmark"):
        """
        Initializes the Constrained Moving Peaks Benchmark generator.

        Args:
            f_params: A dictionary of parameters for the objective landscape
                      (f), to be passed to the MovingPeaksBenchmark constructor.
            g_params: A dictionary of parameters for the constraint landscape
                      (g), to be passed to the MovingPeaksBenchmark constructor.
            problem_name: A name for the problem instance.
        """
        if f_params.get('dimension') != g_params.get('dimension'):
            raise ValueError(
                "The dimensions of the objective (f) and constraint (g) "
                "landscapes must be the same."
            )

        self.f_landscape = MovingPeaksBenchmark(**f_params)
        self.g_landscape = MovingPeaksBenchmark(**g_params)

        # The bounds are determined by the objective function landscape `f`
        self._bounds = self.f_landscape.get_bounds()
        self._name = problem_name
        self._dimension = self.f_landscape.get_dimension()
        self._is_objective_dynamic = self.f_landscape._change_frequency > 0
        self._is_constraint_dynamic = self.g_landscape._change_frequency > 0

    def _h_fitness(self, x: List[float]) -> float:
        """
        The composed objective function, h(x) = f(x) - g(x).
        """
        f_val = self.f_landscape._get_raw_maximization_value(x)
        g_val = self.g_landscape._get_raw_maximization_value(x)
        h_val = f_val - g_val
        return -h_val # Return negative for minimization

    def _constraint(self, x: List[float]) -> float:
        """
        The inequality constraint function, g(x) - f(x) <= 0.
        A solution is feasible if the return value is <= 0.
        """
        f_val = self.f_landscape._get_raw_maximization_value(x)
        g_val = self.g_landscape._get_raw_maximization_value(x)
        return g_val - f_val

    def get_objective_functions(self) -> List[Callable[[List[float]], float]]:
        return [self._h_fitness]

    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        # One inequality constraint, no equality constraints
        return ([self._constraint], [])

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self._bounds

    def get_dimension(self) -> int:
        return self._dimension

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (self._is_objective_dynamic, self._is_constraint_dynamic)

    def change_environment(self, iteration: int) -> None:
        """
        Updates both the objective and constraint landscapes.
        """
        self.f_landscape.change_environment(iteration)
        self.g_landscape.change_environment(iteration)

    def initialize_solution(self) -> List[float]:
        """
        Generates an initial solution within the bounds of the objective 'f'
        landscape.
        """
        return self.f_landscape.initialize_solution()

    @property
    def is_multiobjective(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name