from typing import List, Tuple
from . import Problem
import random

class Sphere(Problem[List[float]]):
    def __init__(self, dimension: int = 2):
        self._dimension = dimension
        self._bounds = ([0.0] * dimension, [10.0] * dimension)
        self._name = "Sphere"

    def evaluate(self, solution: List[float]) -> Tuple[List[float], List[float]]:
        objective = sum(x * x for x in solution)
        constraint_violation = [max(0, sum(solution) - 10)]
        return [objective], constraint_violation

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self._bounds

    def get_dimension(self) -> int:
        return self._dimension

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def update(self, iteration: int) -> None:
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