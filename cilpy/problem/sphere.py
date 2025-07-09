# cilpy/problem/sphere.py

from typing import Callable, List, Tuple
from . import Problem
import random

class Sphere(Problem[List[float]]):
    def __init__(self, dimension: int = 2):
        self._dimension = dimension
        self._bounds = ([-10.0] * dimension, [10.0] * dimension)
        self._name = "Sphere"
        self._change_frequency = 0

    def get_objective_functions(self) -> List[Callable[[List[float]], float]]:
        def objective(x: List[float]) -> float:
            return sum(x_i * x_i for x_i in x)
        return [objective]

    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        return [], []

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self._bounds

    def get_dimension(self) -> int:
        return self._dimension

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def change_environment(self, iteration: int) -> None:
        pass

    def initialize_solution(self) -> List[float]:
        lower, upper = self._bounds
        return [random.uniform(l, u) for l, u in zip(lower, upper)]

    @property
    def is_multiobjective(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name