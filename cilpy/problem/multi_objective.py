# cilpy/problem/multi_objective.py
"""
Multi-Objective Benchmark Optimization Problems.

This module provides implementations of benchmark functions for multi-objective
optimization, adhering to the `Problem` interface.
"""
from typing import List, Tuple

from cilpy.problem import Problem, Evaluation


class SCH1(Problem[List[float], List[float]]):
    """The Schaffer Function N. 1.

    This is a classic, single-variable, two-objective optimization problem.
    It is widely used as a simple benchmark to test the ability of a
    multi-objective algorithm to find a convex Pareto front.

    The objective functions are:
    f1(x) = x^2
    f2(x) = (x - 2)^2

    The Pareto-optimal set corresponds to x in [0, 2].
    """

    def __init__(self, domain: Tuple[float, float] = (-10.0, 10.0)):
        """Initializes a Schaffer Function N. 1 problem instance.

        Args:
            domain (Tuple[float, float], optional): A tuple `(min_val, max_val)`
                defining the search space boundary for the single decision
                variable. Defaults to (-10.0, 10.0).
        """
        # This problem has one dimension (a single 'x' value)
        super().__init__(
            dimension=1,
            bounds=([domain[0]], [domain[1]]),
            name="SCH1"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        """Evaluates the SCH1 function for a given solution.

        Args:
            solution (List[float]): A list containing a single float `[x]`
                representing the decision variable.

        Returns:
            Evaluation[List[float]]: An Evaluation object where the fitness is a
                list of two floats `[f1, f2]`, and there are no constraint
                violations.
        """
        x = solution[0]

        # The two objective values
        f1 = x**2
        f2 = (x - 2)**2

        fitness = [f1, f2]
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the SCH1 function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def get_current_optimum(self) -> Evaluation[List[float]]:
        """
        Returns the Ideal Point of the Pareto Optimal Front.

        For SCH1, f1_min = 0 (at x=0) and f2_min = 0 (at x=2).
        The Ideal Point is therefore [0, 0].
        This represents the theoretical best value for each objective.
        """
        ideal_point = [0.0, 0.0]
        return Evaluation(fitness=ideal_point)

    def get_current_anti_optimum(self) -> Evaluation[List[float]]:
        """
        Returns the Nadir Point of the Pareto Optimal Front.

        For SCH1, the Pareto front exists for x in [0, 2].
        At x=0, the point is [0, 4].
        At x=2, the point is [4, 0].
        The worst value for f1 on the front is 4, and for f2 is 4.
        The Nadir Point is therefore [4, 4].
        """
        nadir_point = [4.0, 4.0]
        return Evaluation(fitness=nadir_point)
