# cilpy/problem/functions.py

# Credit: Axel Thevenot implemented many of these functions
# https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective

import numpy as np
from typing import Callable, List, Tuple

from . import Problem


class Sphere(Problem[np.ndarray, float]):
    name = "Sphere"
    latex_formula = r"f(\mathbf{x})=\sum_{i=1}^{d} x_i^{2}"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-5.12, 5.12], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f(0, ..., 0)=0"
    continuous = True
    convex = True
    separable = True
    differentiable = False
    mutimodal = False
    randomized_term = False
    parametric = False

    def __init__(
        self, dimension: int, bounds: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        self.dimension = dimension
        self.bounds = bounds

    def __call__(self, X):
        X = np.asarray(X)
        return np.sum(X**2)

    def get_objective_functions(self) -> List[Callable[[np.ndarray], float]]:

        return [self.__call__]

    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        return [], []

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.bounds

    def get_param(self) -> dict:
        return {}

    def get_dimension(self) -> int:
        return self.dimension

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (X, self(X))


class Ackley(Problem[np.ndarray, float]):
    name = "Ackley"
    latex_formula = r"f(\mathbf{x}) = -a \cdot exp(-b\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2})-exp(\frac{1}{d}\sum_{i=1}^{d}cos(c \cdot x_i))+ a + exp(1)"
    latex_formula_dimension = r"d \in \mathbb{N}_{+}^{*}"
    latex_formula_input_domain = (
        r"x_i \in [-32, 32], \forall i \in \llbracket 1, d\rrbracket"
    )
    latex_formula_global_minimum = r"f((0, ..., 0)) = 0"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    mutimodal = True
    randomized_term = False
    parametric = True

    def __init__(self,
                 dimension: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 a=20,
                 b=0.2,
                 c=2 * np.pi):
        self.dimension = dimension
        self.bounds = bounds
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, X: np.ndarray) -> float:
        res = -self.a * np.exp(-self.b * np.sqrt(np.mean(X**2)))
        res = res - np.exp(np.mean(np.cos(self.c * X))) + self.a + np.exp(1)
        return res

    def get_objective_functions(self) -> List[Callable[[np.ndarray], float]]:

        return [self.__call__]

    def get_constraint_functions(self) -> Tuple[List[Callable], List[Callable]]:
        return [], []

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.bounds

    def get_param(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c}

    def get_dimension(self) -> int:
        return self.dimension

    def get_global_minimum(self, d):
        X = np.array([1 / (i + 1) for i in range(d)])
        X = np.array([0 for _ in range(d)])
        return (X, self(X))

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)



if __name__ == "__main__":
    lower = np.array([-5.12, -5.12])
    upper = np.array([5.12, 5.12])
    my_sphere = Sphere(2, (lower, upper))
    print(my_sphere([2, 2]))

    # my_sphere = Sphere(2, [np.ndarray([-5.12, -5.12]), np.ndarray([-5.12, -5.12])])
    # print(my_sphere([2, 2]))
