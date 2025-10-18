# cilpy/problem/mpb.py
"""
The Moving Peaks Benchmark (MPB) for dynamic optimization problems.

This module provides an implementation of the Moving Peaks Benchmark (MPB)
generator, which produces dynamic, multi-peaked, unconstrained optimization
landscapes. It is designed to be a maximization problem, but this implementation
negates the fitness to allow for minimization-based solvers.
"""

import random
from typing import List, Tuple

import numpy as np

from cilpy.problem import Evaluation, Problem


class _Peak:
    """Represents a single peak within the Moving Peaks Benchmark landscape.

    This is a helper class that encapsulates the state and behavior of one peak,
    including its position, height, width, and how it changes over time.

    Attributes:
        v (np.ndarray): The vector representing the peak's location (center).
        h (float): The scalar value for the peak's height.
        w (float): The scalar value for the peak's width.
        s_v (np.ndarray): The shift vector, influencing the peak's movement.
    """

    def __init__(self, position: np.ndarray, height: float, width: float):
        """Initializes a _Peak instance.

        Args:
            position (np.ndarray): The initial location vector of the peak.
            height (float): The initial height of the peak.
            width (float): The initial width of the peak.
        """
        self.v = position
        self.h = height
        self.w = width
        self.s_v = np.zeros_like(position)

    def evaluate(self, x: np.ndarray) -> float:
        """Calculates the peak's value at a given position `x`.

        The peak function is defined as: p_i(x) = h - w * ||x - v||, where
        ||.|| is the Euclidean distance.

        Args:
            x (np.ndarray): The candidate solution's position vector.

        Returns:
            float: The value of the peak function at position `x`.
        """
        dist = np.linalg.norm(x - self.v)
        return float(self.h - self.w * dist)

    def update(
        self,
        height_sev: float,
        width_sev: float,
        change_sev: float,
        lambda_param: float,
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        """Updates the peak's parameters for the next environment.

        This method implements the recurrence relation described in Equation 4.4
        of Gary Pamparà's PhD thesis to modify the peak's height, width, and
        position.

        Args:
            height_sev (float): The severity of height changes.
            width_sev (float): The severity of width changes.
            change_sev (float): The severity of positional changes (shift length).
            lambda_param (float): A correlation coefficient for peak movement.
                A value of 0.0 implies fully random movement.
            bounds (Tuple[np.ndarray, np.ndarray]): The problem's lower and
                upper boundaries for enforcing constraints.
        """
        dim = len(self.v)
        # Generate a random vector p_r and normalize to length 'change_sev'
        p_r = np.random.uniform(-1, 1, size=dim)
        mag_pr = np.linalg.norm(p_r)
        if mag_pr > 0:
            p_r *= change_sev / mag_pr

        # Calculate new shift vector based on Equation 4.4
        combined_move = (1.0 - lambda_param) * p_r + lambda_param * self.s_v
        mag_combined = np.linalg.norm(combined_move)
        if mag_combined > 0:
            self.s_v = combined_move * (change_sev / mag_combined)
        else:
            self.s_v = np.zeros(dim)

        # Update height and width with Gaussian noise
        self.h += height_sev * np.random.normal(0, 1)
        self.w += width_sev * np.random.normal(0, 1)

        # Update position
        self.v += self.s_v

        # Enforce boundary conditions via reflection
        min_b, max_b = bounds
        low_mask = self.v < min_b
        self.v[low_mask] = 2 * min_b[low_mask] - self.v[low_mask]
        self.s_v[low_mask] *= -1.0
        high_mask = self.v > max_b
        self.v[high_mask] = 2 * max_b[high_mask] - self.v[high_mask]
        self.s_v[high_mask] *= -1.0


class MovingPeaksBenchmark(Problem[np.ndarray, float]):
    """An implementation of the Moving Peaks Benchmark (MPB) generator.

    This class conforms to the `Problem` interface and produces dynamic,
    unconstrained optimization problems. The objective is to find the maximum
    value in a landscape composed of several moving peaks.

    Note:
        Since most solvers are minimizers, the `evaluate` method returns the
        *negated* value of the MPB function. Minimizing this value is
        equivalent to maximizing the original function.

    Attributes:
        peaks (List[_Peak]): A list of the peak objects in the landscape.
    """

    def __init__(
        self,
        dimension: int = 2,
        num_peaks: int = 10,
        domain: Tuple[float, float] = (0.0, 100.0),
        min_height: float = 30.0,
        max_height: float = 70.0,
        min_width: float = 1.0,
        max_width: float = 12.0,
        change_frequency: int = 5000,
        change_severity: float = 1.0,
        height_severity: float = 7.0,
        width_severity: float = 1.0,
        lambda_param: float = 0.0,
        name: str = "MovingPeaksBenchmark",
    ):
        """Initializes the Moving Peaks Benchmark problem.

        Args:
            dimension (int): The dimensionality of the search landscape.
            num_peaks (int): The number of peaks in the landscape.
            domain (Tuple[float, float]): The `(min, max)` coordinates for the
                symmetric search space.
            min_height (float): The minimum initial height of a peak.
            max_height (float): The maximum initial height of a peak.
            min_width (float): The minimum initial width of a peak.
            max_width (float): The maximum initial width of a peak.
            change_frequency (int): The number of evaluations between landscape changes.
            change_severity (float): Controls how severely peak positions change.
            height_severity (float): Controls how severely peak heights change.
            width_severity (float): Controls how severely peak widths change.
            lambda_param (float): Correlates peak movement over time. A value of
                0.0 results in random movement direction at each change.
            name (str): The name of the problem instance.
        """
        min_bounds = np.array([domain[0]] * dimension)
        max_bounds = np.array([domain[1]] * dimension)
        super().__init__(dimension, (min_bounds, max_bounds), name)

        self._change_frequency = change_frequency
        self._change_sev = change_severity
        self._height_sev = height_severity
        self._width_sev = width_severity
        self._lambda = lambda_param

        self.peaks: List[_Peak] = []
        for _ in range(num_peaks):
            pos = np.random.uniform(domain[0], domain[1], size=dimension)
            height = random.uniform(min_height, max_height)
            width = random.uniform(min_width, max_width)
            self.peaks.append(_Peak(pos, height, width))

        self._base_value = 0.0  # As per Equation 4.2
        self._eval_count = 0

    def evaluate(self, solution: np.ndarray) -> Evaluation[float]:
        """Evaluates a solution and returns its fitness.

        This method checks if the environment should change based on the
        evaluation count. It then calculates the function value as the maximum
        of all peak evaluations.

        Args:
            solution (np.ndarray): The candidate solution to be evaluated.

        Returns:
            Evaluation[float]: An Evaluation object containing the negated
                fitness value for use with minimization solvers.
        """
        self._eval_count += 1
        if self._change_frequency > 0 and self._eval_count > 0 and \
           self._eval_count % self._change_frequency == 0:
            for peak in self.peaks:
                peak.update(
                    height_sev=self._height_sev,
                    width_sev=self._width_sev,
                    change_sev=self._change_sev,
                    lambda_param=self._lambda,
                    bounds=self.bounds,
                )

        peak_values = [p.evaluate(solution) for p in self.peaks]
        max_value = float(max([self._base_value] + peak_values))

        # Negate the value for minimization solvers
        return Evaluation(fitness=-max_value)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the problem's objectives are dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(True, False)` as the objective
                function changes over time but there are no constraints.
        """
        return (True, False)

if __name__ == "__main__":
    def demonstrate_mpb(name: str, lambda_param: float):
        """Helper function to run and print a scenario."""
        print("-" * 50)
        print(f"Demonstration: {name}")
        print(f"Using lambda_param = {lambda_param}")
        print("-" * 50)

        # Common parameters for the demonstration
        params = {
            "dimension": 2,
            "num_peaks": 5,
            "change_frequency": 10,  # Change environment every 10 evaluations
            "lambda_param": lambda_param,
        }

        # Instantiate the problem
        problem = MovingPeaksBenchmark(**params)

        # We will track the position and value of a single peak to see how it moves.
        tracked_peak_index = 0
        
        # We will also evaluate a static point to see how the landscape changes underneath it.
        static_point_to_test = np.array([50.0, 50.0])

        num_changes_to_observe = 5
        total_evaluations = params["change_frequency"] * num_changes_to_observe

        for i in range(total_evaluations + 1):
            # The actual evaluation triggers the internal counter
            evaluation = problem.evaluate(static_point_to_test)

            # Check if the environment just changed
            if i > 0 and i % params["change_frequency"] == 0:
                change_num = i // params["change_frequency"]
                peak_pos = problem.peaks[tracked_peak_index].v
                
                print(f"\nEnvironment Change #{change_num} (at evaluation {i}):")
                print(f"  - Position of Peak {tracked_peak_index}: [{peak_pos[0]:.2f}, {peak_pos[1]:.2f}]")
                print(f"  - Value at static point [50, 50]: {-evaluation.fitness:.2f}")

        print("\n")


    # --- Scenario 1: Random Peak Movement ---
    # With lambda_param = 0.0, the movement direction is completely random at
    # each change.
    demonstrate_mpb(name="Random Peak Movement", lambda_param=0.0)

    # --- Scenario 2: Linear (Correlated) Peak Movement ---
    # With lambda_param = 1.0, the movement is strongly correlated with the
    # previous direction, resulting in more linear paths.
    demonstrate_mpb(name="Linear/Correlated Peak Movement", lambda_param=1.0)
