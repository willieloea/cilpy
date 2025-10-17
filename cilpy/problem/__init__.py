"""
The problem module.
"""
# cilpy/problem/__init__.py

from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Tuple, TypeVar

# Generic types for candidate solutions and fitness values
SolutionType = TypeVar("SolutionType")
FitnessType = TypeVar("FitnessType")


class Problem(ABC, Generic[SolutionType, FitnessType]):
    """An abstract interface for optimization problems in cilpy.

    All problems in `cilpy.problem` should implement this interface to ensure
    compatibility with solvers and comparison tools. The interface is generic
    to support different solution representations (e.g., List[float], List[int],
    custom objects).

    Attributes:
        name (str): A string containing the name of the problem.
        dimension (int): An integer count of the dimension of the problem landscape.
        bounds (Tuple[SolutionType, SolutionType]): Search space boundaries for the problem
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

        Returning a List of callables is a way to accommodate both single- and
        multi-objective problems within the same interface.
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
    def is_dynamic(self) -> Tuple[bool, bool]:
        """
        Indicates whether the problem has dynamic objectives or constraints.

        Returns:
            Tuple[bool, bool]: A tuple of (is_objective_dynamic,
                               is_constraint_dynamic).
        """
        pass
