
    # def get_current_optimum(self) -> Evaluation[float]:
    #     """
    #     Returns the evaluation of the true global optimum for the current state
    #     of the problem landscape.

    #     For dynamic problems, this value may change over time. For static
    #     problems, it will be constant.
    #     """
    #     return Evaluation(fitness=0)

    # def get_current_anti_optimum(self) -> Evaluation[float]:
    #     """
    #     Returns the evaluation of the true global anti-optimum (worst possible
    #     value) for the current state of the problem landscape.
    #     """
    #     return Evaluation(fitness=0)
# cilpy/problem/unconstrained.py
"""
Benchmark Optimization Functions.

This module provides concrete implementations of common benchmark functions
for single-objective, unconstrained optimization, adhering to the `Problem`
interface.
"""
import math
from typing import List, Tuple

from cilpy.problem import Problem, Evaluation

class Sphere(Problem[List[float], float]):
    """The Sphere function.

    A continuous, convex, and unimodal benchmark function. It is one of the
    simplest benchmark problems. The global minimum is at the origin (0, ..., 0)
    with a fitness value of 0.

    The function is defined as: f(x) = Sum(x_i^2) for i = 1 to n.
    """

    def __init__(self, dimension: int, domain: Tuple[float, float] = (-5.12, 5.12)):
        """Initializes a Sphere function problem instance.

        Args:
            dimension (int): The number of decision variables (dimensions).
            domain (Tuple[float, float], optional): A tuple `(min_val, max_val)`
                defining the symmetric search space boundary for each dimension.
                Defaults to (-5.12, 5.12).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension,
            bounds=(lower_bounds, upper_bounds),
            name="Sphere"
        )
        # Pre-calculate the anti-optimum
        bound_mag = max(abs(domain[0]), abs(domain[1]))
        self._anti_optimum_fitness = self.dimension * (bound_mag ** 2)

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Sphere function for a given solution.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness
                (a single float) and no constraint violations.
        """
        fitness = sum(x**2 for x in solution)
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Sphere function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def get_current_optimum(self) -> Evaluation[float]:
        """Returns the global optimum of the Sphere function, which is 0."""
        return Evaluation(fitness=0.0)

    def get_current_anti_optimum(self) -> Evaluation[float]:
        """
        Returns the global anti-optimum (max value) of the Sphere function.
        This occurs at the corners of the domain bounds.
        """
        return Evaluation(fitness=self._anti_optimum_fitness)


class Quadratic(Problem[List[float], float]):
    """The Quadratic function.

    Also known as the Schwefel 1.2 or Double Sum function. It is a continuous,
    convex, and unimodal benchmark function. It is more difficult than the
    Sphere  function because the variables are dependent on each other due to
    the cumulative summation.

    The global minimum is at the origin (0, ..., 0) with a fitness value of 0.

    The function is defined as: f(x) = Sum_{j=1}^n ( Sum_{k=1}^j x_k )^2
    """

    def __init__(
            self,
            dimension: int,
            domain: Tuple[float, float] = (-100.0, 100.0)
        ):
        """Initializes a Quadric function problem instance.

        Args:
            dimension (int): The number of decision variables (dimensions).
            domain (Tuple[float, float], optional): A tuple `(min_val, max_val)`
                defining the symmetric search space boundary for each dimension.
                Defaults to (-100.0, 100.0) based on standard benchmarks.
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension,
            bounds=(lower_bounds, upper_bounds),
            name="Quadric"
        )
        # Pre-calculate anti-optimum
        # Max value is when all x_i are at the boundary with the largest magnitude
        bound_mag = max(abs(domain[0]), abs(domain[1]))
        n = self.dimension
        # The sum is (bound_mag^2) * (1^2 + 2^2 + ... + n^2)
        # Sum of squares formula: n(n+1)(2n+1)/6
        sum_of_squares = n * (n + 1) * (2 * n + 1) / 6
        self._anti_optimum_fitness = (bound_mag ** 2) * sum_of_squares

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Quadric function for a given solution.

        Implements f(x) = Σ(Σ(x_k)^2) efficiently using a running sum.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness.
        """
        fitness = 0.0
        cumulative_sum = 0.0
        for x in solution:
            cumulative_sum += x
            fitness += cumulative_sum ** 2

        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Quadric function is not dynamic.

        Returns:
            Tuple[bool, bool]: (False, False).
        """
        return (False, False)

    def get_current_optimum(self) -> Evaluation[float]:
        """Returns the global optimum of the Quadric function, which is 0."""
        return Evaluation(fitness=0.0)

    def get_current_anti_optimum(self) -> Evaluation[float]:
        """
        Returns the global anti-optimum of the Quadric function, occurring when
        all variables are at the same domain boundary.
        """
        return Evaluation(fitness=self._anti_optimum_fitness)


class Ackley(Problem[List[float], float]):
    """The Ackley function.

    A widely used multimodal benchmark function characterized by a nearly flat
    outer region and a large number of local minima. Its global minimum is at
    the origin (0, ..., 0) with a fitness value of 0.

    The function is defined as:
    f(x) = -a * exp(-b * sqrt(1/n * Sum(x_i^2))) - exp(1/n * Sum(cos(c*x_i))) + a + exp(1)
    """

    def __init__(self, dimension: int, domain: Tuple[float, float] = (-32.768, 32.768)):
        """Initializes an Ackley function problem instance.

        The standard parameters `a=20`, `b=0.2`, and `c=2pi` are used.

        Args:
            dimension (int): The number of decision variables (dimensions).
            domain (Tuple[float, float], optional): A tuple `(min_val, max_val)`
                defining the symmetric search space boundary for each dimension.
                Defaults to (-32.768, 32.768).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension,
            bounds=(lower_bounds, upper_bounds),
            name="Ackley"
        )
        # Standard parameters for the Ackley function
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Ackley function for a given solution.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness
                (a single float) and no constraint violations.
        """
        sum_sq = sum(x**2 for x in solution)
        sum_cos = sum(math.cos(self.c * x) for x in solution)

        term1 = -self.a * math.exp(-self.b * math.sqrt(sum_sq / self.dimension))
        term2 = -math.exp(sum_cos / self.dimension)

        fitness = term1 + term2 + self.a + math.e
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Ackley function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def get_current_optimum(self) -> Evaluation[float]:
        """Returns the global optimum of the Ackley function, which is 0."""
        return Evaluation(fitness=0.0)

    def get_current_anti_optimum(self) -> Evaluation[float]:
        """
        Returns an approximation of the global anti-optimum of the Ackley
        function. The value approaches `a + e` in the flat outer regions.
        """
        # A common and effective approximation for the anti-optimum
        return Evaluation(fitness=(self.a + math.e))


if __name__ == "__main__":
    my_sphere = Sphere(2, (-5.12, 5.12))
    print(my_sphere.evaluate([0, 0]))

    my_quadratic = Quadratic(2, (-5.12, 5.12))
    print(my_quadratic.evaluate([0, 0]))

    my_ackley = Ackley(2, (-5.12, 5.12))
    print(my_ackley.evaluate([0, 0]))
