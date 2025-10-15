"""
The problem module.
"""
# cilpy/problem/__init__.py

from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Tuple, TypeVar

# Generic types for solutions and fitness values
SolutionType = TypeVar("SolutionType")
FitnessType = TypeVar("FitnessType")


class Problem(ABC, Generic[SolutionType, FitnessType]):
    """
    An abstract interface for optimization problems in cilpy.

    All problems in `cilpy.problem` should implement this interface to ensure
    compatibility with solvers and comparison tools. The interface is generic
    to support different solution representations (e.g., List[float], List[int],
    custom objects).
    """

    @abstractmethod
    def __init__(self,
                 dimension: int,
                 bounds: Tuple[SolutionType, SolutionType],
                 name: str) -> None:
        self.name = name
        self.dimension = dimension
        self.bounds = bounds

    @abstractmethod
    def get_objective_functions(self) -> List[Callable[[SolutionType], FitnessType]]:
        """
        Returns the objective function(s) of a problem.
        """
        pass

    @abstractmethod
    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        """
        Returns the constraint functions of a problem.

        Returns:
            Tuple[List[Callable], List[Callable]]: A tuple containing:
                - List of inequality constraints
                - List of equality constraints
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[SolutionType, SolutionType]:
        """
        Returns the search space boundaries or constraints for the problem.

        Returns:
            Any: The bounds or constraints defining the solution space (e.g.,
            [List[float], List[float]] for real-valued problems, or other
            structures for different solution types).
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Returns the dimensionality or size of the solution space.

        Returns:
            int: The number of dimensions in a solution
        """
        pass

    @abstractmethod
    def is_dynamic(self) -> Tuple[bool, bool]:
        """
        Indicates whether the problem has dynamic objectives or constraints.

        Returns:
            Tuple[bool, bool]: A tuple of (is_objective_dynamic,
                               is_constraint_dynamic).
        """
        pass
