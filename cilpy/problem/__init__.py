# cilpy/problem/__init__.py
"""
The problem module.

This module defines the abstract interface for optimization problems within
the cilpy library, along with a dataclass for encapsulating evaluation results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

# Generic types for candidate solutions and fitness values
SolutionType = TypeVar("SolutionType")
FitnessType = TypeVar("FitnessType")


@dataclass
class Evaluation(Generic[FitnessType]):
    """A container for the results of a problem evaluation.

    Attributes:
        fitness (FitnessType): The objective function value(s) of the solution.
            This can be a single float for single-objective problems or a list
            of floats for multi-/many-objective problems.
        constraints_inequality (Optional[List[float]]): A list of values
            representing the inequality constraint violations.
            Each value `g(x)` should be <= 0 for a feasible solution.
            A positive value indicates the degree of violation.
            Defaults to `None` if no inequality constraints exist.
        constraints_equality (Optional[List[float]]): A list of values
            representing the equality constraint violations.
            Each value `h(x)` should be == 0 for a feasible solution.
            A non-zero value indicates the degree of violation.
            Defaults to `None` if no equality constraints exist.
    """
    fitness: FitnessType
    constraints_inequality: Optional[List[float]] = None
    constraints_equality: Optional[List[float]] = None


class Problem(ABC, Generic[SolutionType, FitnessType]):
    """An abstract interface for optimization problems in cilpy.

    All problems in `cilpy.problem` should implement this interface to ensure
    compatibility with solvers and comparison tools. The interface is generic
    to support different solution representations (e.g., List[float], List[int],
    custom objects).

    Attributes:
        name (str): A string containing the name of the problem.
        dimension (int): An integer count of the dimension of the problem landscape.
        bounds (Tuple[SolutionType, SolutionType]): Search space boundaries for
            the problem, typically a tuple `(lower_bounds, upper_bounds)`.
    """
    @abstractmethod
    def __init__(self,
                 dimension: int,
                 bounds: Tuple[SolutionType, SolutionType],
                 name: str) -> None:
        """Initializes a Problem instance.

        Args:
            dimension (int): The dimensionality or size of the solution space.
            bounds (Tuple[SolutionType, SolutionType]): The search space
                boundaries or constraints for the problem, defining the feasible
                range for each decision variable. For real-valued problems, this
                might be
                `([min_val_d1, ..., min_val_dn], [max_val_d1, ..., max_val_dn])`.
            name (str): The name of the optimization problem.
        """

        self.name = name
        self.dimension = dimension
        self.bounds = bounds

    @abstractmethod
    def evaluate(self, solution: SolutionType) -> Evaluation[FitnessType]:
        """
        Evaluates a given solution and returns its fitness and constraint
        violations.

        This method provides all necessary information about a solution's
        performance and feasibility for a problem.

        Args:
            solution (SolutionType): The candidate solution to be evaluated.
                Its type depends on the problem (e.g., `List[float]`, `List[int]`).

        Returns:
            Evaluation[FitnessType]: An `Evaluation` object containing the
                solution's fitness value(s) and any constraint violation
                information.
        """
        pass

    @abstractmethod
    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates whether the problem has dynamic objectives or constraints.

        Dynamic problems change over time, requiring solvers to potentially
        re-evaluate stored best solutions or adapt to the changing landscape.

        Returns:
            Tuple[bool, bool]: A tuple where:
                - The first boolean (`is_objective_dynamic`) is True if the
                  objective function(s) change over time, False otherwise.
                - The second boolean (`is_constraint_dynamic`) is True if the
                  constraint functions change over time, False otherwise.
        """
        pass
