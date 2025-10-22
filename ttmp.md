This is what is in my mkdocs.yml file, and it describes what I want to include in the various pages:
```
nav:
  - 'Home': 'index.md'             # overview of cilpy and documentation
  - 'Quickstart': 'quickstart.md'  # quick guide on how to implement interfaces and run experiment.
  - 'Included':
    - 'Overview': 'lib/index.md'   # high level description of library components, and how they interact. Also describe the cooperative co-evolutionary Lagrangian solver.
    - 'Problems': 'lib/problem.md' # documentation on problems included in the library.
    - 'Solvers': 'lib/solver.md'   # documentation on solvers included in the library.
    - 'Compare': 'lib/compare.md'  # documentation on comparison included tools in the library.
  - 'API Reference':
    - 'Overview': 'api/index.md'   # core interfaces of the library (problem, solver, and compare)
    - 'Problem': 'api/problem.md'
    - 'Solver': 'api/solver.md'
    - 'Compare': 'api/compare.md'
  - 'Developers':
    - 'Intro': 'dev/index.md'      # developer guide and contribution rules
```

For context, I will show you examples of problems, solvers, and how they are
used together.

This is an example of a problem:
```
# cilpy/problem/mpb.py
"""
The Moving Peaks Benchmark (MPB) for dynamic optimization problems.

This module provides an implementation of the Moving Peaks Benchmark (MPB)
generator, a widely used tool for creating dynamic, multi-peaked optimization
landscapes. It is designed to test the ability of optimization algorithms to
adapt to changing environments.

The MPB landscape is defined by a number of peaks, each with its own height,
width, and position. At specified intervals (controlled by `change_frequency`),
these peak properties are updated, causing the landscape to shift, change shape,
or both.

While the benchmark is naturally a maximization problem, this implementation
negates the fitness value upon evaluation, allowing it to be used directly with
standard minimization algorithms.

--------------------------------------------------------------------------------
The 28 Standard Problem Classes & The `generate_mpb_configs` Function
--------------------------------------------------------------------------------
This module also includes the `generate_mpb_configs` helper function, which
programmatically creates parameter dictionaries for the 28 standard problem
classes defined by Duhain and Engelbrecht.

These classes are identified by a 3-letter acronym (e.g., 'A1L'), which combines
one code from each of the three classification schemes detailed below.

### 1. Duhain & Engelbrecht: Spatial and Temporal Severity (First Letter)
Defines the magnitude and frequency of changes.

-   **Progressive ('P')**: Frequent, small changes.
    -   `change_frequency`: Low value (high temporal change).
    -   `change_severity` (`s`): Low value.
    -   `height_severity`: Low value.

-   **Abrupt ('A')**: Infrequent, large changes.
    -   `change_frequency`: High value (low temporal change).
    -   `change_severity` (`s`): High value.
    -   `height_severity`: High value.

-   **Chaotic ('C')**: Frequent, large changes.
    -   `change_frequency`: Low value (high temporal change).
    -   `change_severity` (`s`): High value.
    -   `height_severity`: High value.

### 2. Hu & Eberhart / Shi & Eberhart: Optima Modification (Second Letter)
Defines *what* changes about the peaks (position, value, or both).

-   **Type I ('1')**: Locations change, heights are constant.
    -   `height_severity`: Set to 0.0.
    -   Requires `change_severity` (`s`) != 0 for movement.

-   **Type II ('2')**: Locations are static, heights change.
    -   `height_severity`: Set to a non-zero value.
    -   Requires `change_severity` (`s`) = 0 to prevent movement.

-   **Type III ('3')**: Both locations and heights change.
    -   `height_severity`: Set to a non-zero value.
    -   Requires `change_severity` (`s`) != 0 for movement.

### 3. Angeline: Optima Trajectory (Third Letter)
Defines the *pattern* of peak movement.

-   **Linear ('L')**: Peaks move in a straight, correlated line.
    -   `lambda_param`: Set to 1.0.
    -   Requires `change_severity` (`s`) != 0 for movement.

-   **Circular ('C')**: Peaks have a periodic movement pattern. This is achieved in
    the parameterization by preventing translational movement.
    -   Requires `change_severity` (`s`) = 0.

-   **Random ('R')**: Peaks move randomly without a discernible pattern.
    -   `lambda_param`: Set to 0.0.
    -   Requires `change_severity` (`s`) != 0 for movement.

### Conflict Resolution
Some combinations are impossible (e.g., a Type II problem, which requires
`s = 0`, cannot have Linear movement, which requires `s != 0`). The
`generate_mpb_configs` function marks these impossible configurations by setting
the `change_severity` parameter to the string 'XXX'.
"""

import itertools
import numpy as np
import random

from typing import Any, Dict, List, Tuple

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
        # Update height with Gaussian noise
        self.h += height_sev * np.random.normal(0, 1)

        # Update width with Gaussian noise
        self.w += width_sev * np.random.normal(0, 1)

        # Update shift vector according to equation 4.4
        dim = len(self.v)
        # Generate a random vector p_r and normalize to length 'change_sev'
        p_r = np.random.uniform(-1, 1, size=dim)
        mag_pr = np.linalg.norm(p_r)
        if mag_pr > 0:
            p_r *= change_sev / mag_pr

        combined_move = (1.0 - lambda_param) * p_r + lambda_param * self.s_v
        mag_move = np.linalg.norm(p_r + self.s_v)
        if mag_move > 0:
            self.s_v = (change_sev / mag_move) * combined_move
        else:
            self.s_v = np.zeros(dim)


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


def generate_mpb_configs(
        dimension: int = 5,
        num_peaks: int = 10,
        domain: Tuple[float, float] = (0.0, 100.0),
        min_height: float = 30.0,
        max_height: float = 70.0,
        min_width: float = 1.0,
        max_width: float = 12.0,
        width_severity: float = 0.05,
        s_for_random: float = 1.0 # s value for s != 0
        ) -> Dict[str, Dict[str, Any]]:
    """
    Programmatically generates parameter dictionaries for all 28 MPB classes.

    This function combines the rules from three classification schemes to generate
    27 dynamic problem configurations and 1 static configuration. It handles
    contradictions between rules as specified.

    Args:
        s_for_random (float): The non-zero value to use for the change_severity
            parameter `s` when a non-zero value is required. Defaults to 1.0.

    Returns:
        Dict[str, Dict]: A dictionary where keys are the 3-letter acronyms
            (e.g., "A1C", "P3L") and values are the corresponding parameter
            dictionaries for the MovingPeaksBenchmark constructor. A "STA" key
            is included for the static case.
    """
    if s_for_random == 0:
        raise ValueError("'s_for_random' must be a non-zero value.")

    # 1. Base Configuration (common to all classes)
    base_params = {
        "dimension": dimension,
        "num_peaks": num_peaks,
        "domain": domain,
        "min_height": min_height,
        "max_height": max_height,
        "min_width": min_width,
        "max_width": max_width,
        "width_severity": width_severity, # Often kept low
    }

    # 2. Define "Low" vs. "High" Values for severity and frequency
    LOW_S, HIGH_S = s_for_random, 10.0
    LOW_H, HIGH_H = 7.0, 15.0

    # High temporal frequency = low number of evaluations between changes
    FREQ_PROGRESSIVE = 1000
    FREQ_ABRUPT = 5000
    FREQ_CHAOTIC = 1000

    # 3. Classification Rules
    # Duhain & Engelbrecht: Severity (Spatial & Temporal)
    # Acronyms: P (Progressive), A (Abrupt), C (Chaotic)
    SEVERITY_CLASSES = {
        'P': {
            "change_severity": LOW_S, "height_severity": LOW_H, 
            "change_frequency": FREQ_PROGRESSIVE,
        },
        'A': {
            "change_severity": HIGH_S, "height_severity": HIGH_H,
            "change_frequency": FREQ_ABRUPT,
        },
        'C': {
            "change_severity": HIGH_S, "height_severity": HIGH_H,
            "change_frequency": FREQ_CHAOTIC,
        },
    }

    # Hu & Eberhart / Shi & Eberhart: Optima Modification
    # Acronyms: 1 (Type I), 2 (Type II), 3 (Type III)
    # We use 's_req' to define the requirement for the change_severity (s)
    MODIFICATION_CLASSES = {
        '1': {"height_severity": 0.0, "s_req": "!=0"},
        '2': {"s_req": "=0"},  # height_severity will be taken from SEVERITY_CLASSES
        '3': {"s_req": "!=0"}, # height_severity will be taken from SEVERITY_CLASSES
    }

    # Angeline: Optima Trajectory
    # Acronyms: L (Linear), C (Circular), R (Random)
    MOVEMENT_CLASSES = {
        'L': {"lambda_param": 1.0, "s_req": "!=0"},
        'C': {"lambda_param": 0.0, "s_req": "=0"}, # lambda is irrelevant when s=0
        'R': {"lambda_param": 0.0, "s_req": "!=0"},
    }

    # --- Generation Logic ---
    all_configs = {}

    # 4. Iterate through all 3*3*3 = 27 combinations
    severity_codes = SEVERITY_CLASSES.keys()
    modification_codes = MODIFICATION_CLASSES.keys()
    movement_codes = MOVEMENT_CLASSES.keys()

    for sev_code, mod_code, mov_code in itertools.product(severity_codes, modification_codes, movement_codes):
        acronym = f"{sev_code}{mod_code}{mov_code}"
        
        # Start with base and add severity parameters
        config = base_params.copy()
        config.update(SEVERITY_CLASSES[sev_code])
        config["name"] = acronym
        
        mod_rules = MODIFICATION_CLASSES[mod_code]
        mov_rules = MOVEMENT_CLASSES[mov_code]
        
        # 5. Resolve Conflicts for `change_severity` (s)
        s_req_mod = mod_rules['s_req']
        s_req_mov = mov_rules['s_req']
        
        is_conflict = (s_req_mod == "!=0" and s_req_mov == "=0") or \
                      (s_req_mod == "=0" and s_req_mov == "!=0")

        if is_conflict:
            # *2L/*2R: 2 requires s = 0, but L&R requires s != 0
            #          2 gets priority since *3* requires s != 0
            if (mod_code == '2' and (mov_code == 'L' or mov_code == 'R')):
                config['change_severity'] = 0.0
            # *1C/*3C: C requires s = 0, but 1&3 requires s != 0
            #          C gets priority since *2* requires s = 0
            elif (mov_code == 'C' and (mod_code == '1' or mod_code == '3')):
                config['change_severity'] = 1.0
            else:
                # This is not supposed to happen, but is kept for clarity
                config['change_severity'] = 'XXX' # Assign placeholder on conflict
        elif s_req_mod == "=0" or s_req_mov == "=0":
            # If either requires s=0 and there's no conflict, it must be 0
            config['change_severity'] = 0.0
        else:
            # Otherwise, s must be non-zero. Use the value from the severity class.
            # This is already set, but we make it explicit for clarity.
            pass

        # 6. Apply overrides from modification and movement rules
        # C1*: C requires hSeverity high, but 1 requires hSeverity = 0
        #      1 (mod rule) gets priority since *2*/*3* requires hSeverity != 0
        if 'height_severity' in mod_rules:
            config['height_severity'] = mod_rules['height_severity']

        if 'lambda_param' in mov_rules:
            config['lambda_param'] = mov_rules['lambda_param']

        all_configs[acronym] = config

    # Add the static problem class
    static_config = base_params.copy()
    static_config.update({
        "change_frequency": 0,
        "change_severity": 0,
        "height_severity": 0,
        "width_severity": 0,
        "lambda_param": 0,
        "name": "STA"
    })
    all_configs["STA"] = static_config

    return all_configs

if __name__ == "__main__":
    def demonstrate_mpb(params: dict):
        """Helper function to run and print a scenario."""
        print("-" * 50)
        print(f"Demonstration: {params.get('name')}")
        print("-" * 50)

        # Instantiate the problem
        problem = MovingPeaksBenchmark(**params)

        # We will track the position and value of a single peak to see how it moves.
        tracked_peak_index = 0

        # We will also evaluate a static point to see how the landscape changes underneath it.
        static_point_to_test = np.array([50.0, 50.0])

        num_changes_to_observe = 5000
        total_evaluations = params["change_frequency"] * num_changes_to_observe

        for i in range(total_evaluations + 1):
            # The actual evaluation triggers the internal counter
            evaluation = problem.evaluate(static_point_to_test)

            # Check if the environment just changed
            if i > 0 and i % (params["change_frequency"]*100) == 0:
                change_num = i // params["change_frequency"]
                peak_pos = problem.peaks[tracked_peak_index].v
                peak_evaluation = problem.evaluate(peak_pos)

                print(f"\nEnvironment Change #{change_num} (at evaluation {i}):")
                print(f"  - Position of Peak {tracked_peak_index}: [{peak_pos[0]:.2f}, {peak_pos[1]:.2f}]")
                print(f"  - Value of Peak {tracked_peak_index}: [{peak_evaluation.fitness}]")
                print(f"  - Value at static point [50, 50]: {-evaluation.fitness:.2f}")

        print("\n")


    # Get all problem configurations
    all_problems = generate_mpb_configs()

    # --- Example ---
    # Select a specific problem, for example, "A2R" (Abrupt, Type II, Random)
    params = all_problems['A2R']
    params["change_frequency"] = 10

    # Instantiate the problem generator
    demonstrate_mpb(params)

    # Parameters can also be specified at creation:
    all_problems = generate_mpb_configs(dimension=2)
    config = all_problems.get('A1C')
    print(f"MPB config for A1C: {config}")

    # And modified:
    config['num_peaks'] = 2 # type: ignore
    print(f"Modified A1C: {config}")

```

This is an example of a solver:
```
# cilpy/solver/ga.py

import random
import copy
from typing import List, Tuple

from ..problem import Problem, Evaluation
from . import Solver


class GA(Solver[List[float], float]):
    """
    A canonical Genetic Algorithm (GA) for single-objective optimization.

    This implementation is based on the structure outlined in Section 3.1.1 of
    Pamparà's PhD thesis. It follows a generational model with selection,
    reproduction (crossover), and mutation operators.

    The algorithm uses:
    - Tournament selection to choose parents.
    - Single-point crossover for reproduction.
    - Gaussian mutation to introduce genetic diversity.
    - Elitism to preserve the best solution across generations.
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 tournament_size: int = 2,
                 **kwargs):
        """
        Initializes the Genetic Algorithm solver.

        Args:
            problem (Problem[List[float], float]): The optimization problem to solve.
            name (str): the name of the solver
            population_size (int): The number of individuals in the population.
            crossover_rate (float): The probability of crossover (pc) occurring
                between two parents.
            mutation_rate (float): The probability of mutation (pm) for each
                gene in an offspring.
            tournament_size (int, optional): The number of individuals to select
                for each tournament. Defaults to 2.
            **kwargs: Additional keyword arguments (not used in this canonical GA).
        """
        super().__init__(problem, name)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        # Initialize population
        self.population = self._initialize_population()
        self.evaluations = [self.problem.evaluate(ind) for ind in self.population]

    def _initialize_population(self) -> List[List[float]]:
        """Creates the initial population with random solutions."""
        population = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(self.population_size):
            individual = [random.uniform(lower_bounds[i], upper_bounds[i])
                          for i in range(self.problem.dimension)]
            population.append(individual)
        return population

    def _selection(self) -> List[List[float]]:
        """Performs tournament selection to choose parents."""
        parents = []
        for _ in range(self.population_size):
            tournament = random.sample(list(range(self.population_size)), self.tournament_size)
            winner_idx = min(tournament, key=lambda i: self.evaluations[i].fitness)
            parents.append(self.population[winner_idx])
        return parents

    def _reproduction(self, parents: List[List[float]]) -> List[List[float]]:
        """Creates offspring through single-point crossover."""
        offspring = []
        for i in range(0, self.population_size, 2):
            p1 = parents[i]
            # Ensure there's a second parent for crossover
            p2 = parents[i + 1] if i + 1 < self.population_size else parents[0]

            # Crossover is only possible if the dimension is > 1.
            if self.problem.dimension > 1 and random.random() < self.crossover_rate:
                crossover_point = random.randint(1, self.problem.dimension - 1)
                c1 = p1[:crossover_point] + p2[crossover_point:]
                c2 = p2[:crossover_point] + p1[crossover_point:]
                offspring.extend([c1, c2])
            else:
                offspring.extend([copy.deepcopy(p1), copy.deepcopy(p2)])
        return offspring[:self.population_size]

    def _mutation(self, offspring: List[List[float]]) -> List[List[float]]:
        """Applies Gaussian mutation to offspring."""
        lower_bounds, upper_bounds = self.problem.bounds
        for individual in offspring:
            for i in range(self.problem.dimension):
                if random.random() < self.mutation_rate:
                    # Add noise from a Gaussian distribution with mean 0
                    mutation_value = random.gauss(0, (upper_bounds[i] - lower_bounds[i]) * 0.1)
                    individual[i] += mutation_value
                    # Clamp the value to within the problem bounds
                    individual[i] = max(lower_bounds[i], min(individual[i], upper_bounds[i]))
        return offspring

    def step(self) -> None:
        """Performs one generation of the Genetic Algorithm."""
        # 1. Selection
        parents = self._selection()

        # 2. Reproduction
        offspring = self._reproduction(parents)

        # 3. Mutation
        mutated_offspring = self._mutation(offspring)

        # 4. Evaluate new offspring
        offspring_evaluations = [self.problem.evaluate(ind) for ind in mutated_offspring]

        # 5. Combine (create next generation) with elitism
        # Find the best individual from the current generation
        best_current_idx = min(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        best_individual = self.population[best_current_idx]
        best_evaluation = self.evaluations[best_current_idx]

        # The new population is the mutated offspring
        self.population = mutated_offspring
        self.evaluations = offspring_evaluations

        # Find the worst individual in the new generation and replace it with the best from the previous
        worst_new_idx = max(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        self.population[worst_new_idx] = best_individual
        self.evaluations[worst_new_idx] = best_evaluation

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """Returns the best solution found in the current population."""
        best_idx = min(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        best_solution = self.population[best_idx]
        best_evaluation = self.evaluations[best_idx]
        return [(best_solution, best_evaluation)]


class HyperMGA(GA):
    """
    A Hyper-mutation Genetic Algorithm (HyperM GA) for dynamic optimization.

    This algorithm extends the canonical GA to adapt to changing environments.
    It detects a change in the problem landscape by monitoring the fitness of
    the best solution. If the fitness degrades, it triggers a "hyper-mutation"
    phase with a significantly higher mutation rate for a fixed period to
    re-introduce diversity.

    This implementation is based on the description in Section 3.2.2 of
    Pamparà's PhD thesis.
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 hyper_mutation_rate: float,
                 hyper_period: int,
                 tournament_size: int = 2,
                 **kwargs):
        """
        Initializes the Hyper-mutation Genetic Algorithm solver.

        Args:
            problem (Problem[List[float], float]): The dynamic optimization
                problem to solve.
            name (str): the name of the solver
            population_size (int): The number of individuals in the population.
            crossover_rate (float): The base probability of crossover.
            mutation_rate (float): The standard mutation rate (pm).
            hyper_mutation_rate (float): The higher mutation rate (p_hyper)
                used when the environment changes.
            hyper_period (int): The number of generations to remain in the
                hyper-mutation state after a change is detected.
            tournament_size (int, optional): The number of individuals for
                tournament selection. Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(problem,
                         name,
                         population_size,
                         crossover_rate,
                         mutation_rate,
                         tournament_size,
                         **kwargs)

        self.hyper_mutation_rate = hyper_mutation_rate
        self.hyper_period = hyper_period

        # State tracking variables
        self.f_best: float = float('inf')
        self.hyper_count = 0
        self.is_hyper_mutation = False
        self._update_best_fitness()

    def _update_best_fitness(self):
        """Updates the tracked best fitness value from the current population."""
        current_best_eval = min(self.evaluations, key=lambda e: e.fitness)
        self.f_best = current_best_eval.fitness

    def step(self) -> None:
        """Performs one generation of the HyperM GA."""
        # --- Change Detection ---
        # Evaluate the current population to get f_test
        self.evaluations = [self.problem.evaluate(ind) for ind in self.population]
        f_test = min(e.fitness for e in self.evaluations)

        # If f_best is undefined or fitness has degraded, trigger hyper-mutation
        if self.f_best == float('inf') or f_test > self.f_best:
            self.is_hyper_mutation = True
            self.hyper_count = 0

        # --- State Management ---
        if self.is_hyper_mutation:
            current_mutation_rate = self.hyper_mutation_rate
            self.hyper_count += 1
            if self.hyper_count > self.hyper_period:
                self.is_hyper_mutation = False
        else:
            current_mutation_rate = self.mutation_rate

        # --- Standard GA Operators ---
        # 1. Selection
        parents = self._selection()

        # 2. Reproduction
        offspring = self._reproduction(parents)

        # 3. Mutation (using the current mutation rate)
        # We temporarily set self.mutation_rate for the _mutation method to use
        original_rate = self.mutation_rate
        self.mutation_rate = current_mutation_rate
        mutated_offspring = self._mutation(offspring)
        self.mutation_rate = original_rate  # Restore original rate

        # 4. Evaluate and Combine (with elitism)
        offspring_evaluations = [self.problem.evaluate(ind) for ind in mutated_offspring]

        best_current_idx = min(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        best_individual = self.population[best_current_idx]
        best_evaluation = self.evaluations[best_current_idx]

        self.population = mutated_offspring
        self.evaluations = offspring_evaluations

        worst_new_idx = max(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        self.population[worst_new_idx] = best_individual
        self.evaluations[worst_new_idx] = best_evaluation

        # Update f_best for the next iteration
        self._update_best_fitness()


class RIGA(GA):
    """
    A Random Immigrants Genetic Algorithm (RIGA) for dynamic optimization.

    This algorithm extends the canonical GA by introducing "random immigrants"
    in each generation to maintain diversity. A fixed percentage of the
    population is replaced by newly generated random individuals, which helps
    the algorithm avoid premature convergence and adapt to changing fitness
    landscapes.

    This implementation is based on the description in Section 3.2.3 and
    Algorithm 3.5 of Pamparà's PhD thesis.
    """

    def __init__(self,
                 problem: Problem[List[float], float],
                 name: str,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 immigrant_rate: float,
                 tournament_size: int = 2,
                 **kwargs):
        """
        Initializes the Random Immigrants Genetic Algorithm solver.

        Args:
            problem (Problem[List[float], float]): The dynamic optimization
                problem to solve.
            name (str): the name of the solver
            population_size (int): The number of individuals in the population.
            crossover_rate (float): The probability of crossover.
            mutation_rate (float): The probability of mutation.
            immigrant_rate (float): The proportion of the population to be
                replaced by random immigrants in each generation (p_im).
            tournament_size (int, optional): The number of individuals for
                tournament selection. Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(problem,
                         name,
                         population_size,
                         crossover_rate,
                         mutation_rate,
                         tournament_size,
                         **kwargs)
        self.immigrant_rate = immigrant_rate

    def _generate_immigrants(self, num_immigrants: int) -> List[List[float]]:
        """Generates a specified number of new random individuals."""
        immigrants = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(num_immigrants):
            individual = [random.uniform(lower_bounds[i], upper_bounds[i])
                          for i in range(self.problem.dimension)]
            immigrants.append(individual)
        return immigrants

    def step(self) -> None:
        """Performs one generation of the RIGA."""
        # --- Perform a standard GA step first ---
        super().step()

        # --- Introduce Immigrants ---
        num_immigrants = int(self.population_size * self.immigrant_rate)
        if num_immigrants == 0:
            return

        # 1. Generate immigrants
        immigrants = self._generate_immigrants(num_immigrants)
        immigrant_evals = [self.problem.evaluate(ind) for ind in immigrants]

        # 2. Combine with the current population by replacing the worst
        # Find the indices of the `num_immigrants` worst individuals
        sorted_indices = sorted(range(self.population_size), key=lambda i: self.evaluations[i].fitness)
        worst_indices = sorted_indices[-num_immigrants:]

        # Replace them with the new immigrants
        for i, idx in enumerate(worst_indices):
            self.population[idx] = immigrants[i]
            self.evaluations[idx] = immigrant_evals[i]

```

And this is how I implemented the cooperative co-evolutionary Lagrangian solver:
```
# cilpy/solver/ccls.py
"""A co-evolutionary framework for constrained optimization problems.

This module provides the `CoevolutionaryLagrangianSolver`, a meta-solver that
tackles constrained optimization problems by reformulating them using Lagrangian
relaxation. This approach avoids traditional penalty functions by transforming
the problem into an unconstrained min-max optimization task, which is then
solved by two cooperating populations of solvers.

How the Cooperative Co-evolutionary Framework Works:
-----------------------------------------------------
A constrained optimization problem can be stated as:
  Minimize: f(x)
  Subject to: g_i(x) <= 0  (inequality constraints)
              h_j(x) == 0  (equality constraints)

The Lagrangian function combines the objective and constraints into a single
equation:
  L(x, mu, lambda) = f(x) + Sum(mu_i * g_i(x)) + Sum(lambda_j * h_j(x))

Here, mu and lambda are the Lagrangian multipliers. The solution to the
original problem can be found by solving the min-max problem:
  min_x max_{mu,lambda} L(x, mu, lambda)

This framework implements this min-max search using two populations:
1.  An 'Objective Solver' Population: This population searches the original
    problem's solution space (for 'x'). Its goal is to MINIMIZE the Lagrangian
    function, L(x, mu*, lambda*), where the multipliers (mu*, lambda*) are the best ones
    found so far by the other population.
2.  A 'Multiplier Solver' Population: This population searches the space of
    Lagrangian multipliers (for 'mu' and 'lambda'). Its goal is to MAXIMIZE the
    Lagrangian function, L(x*, mu, lambda), where the solution (x*) is the best one
    found so far by the objective population.

At each step, the best individual from each population is used to define the
fitness landscape for the other. This cooperative process simultaneously drives
the solution 'x' towards feasibility and optimality while evolving the
multipliers to appropriately penalize constraint violations.

This implementation acts as a high-level "coordinator" or "meta-solver". It is
configured with two standard solver instances from the library (e.g., two GAs,
two PSOs for CCPSO) which manage the search process for their respective
populations.
"""

from ..problem import Problem, Evaluation, SolutionType
from . import Solver

class _LagrangianMinProblem(Problem):
    """An internal proxy problem for the objective-space solver ('min' swarm).

    This class wraps the original constrained problem, presenting it to the
    objective solver as an unconstrained problem. Its fitness function is the
    Lagrangian L(x, mu*, lambda*), where the multipliers (mu*, lambda*) are fixed for the
    current generation, having been provided by the multiplier swarm.

    Attributes:
        original_problem (Problem): A reference to the user-defined constrained
            problem.
        fixed_multipliers_inequality (List[float]): The best inequality
            multipliers (mu*) from the multiplier swarm, fixed for this evaluation.
        fixed_multipliers_equality (List[float]): The best equality multipliers
            (lambda*) from the multiplier swarm, fixed for this evaluation.
    """
    def __init__(self, original_problem: Problem):
        """Initializes the proxy problem for the objective space.

        Args:
            original_problem (Problem): The original constrained problem instance
                that will be wrapped.
        """
        super().__init__(original_problem.dimension, original_problem.bounds, "LagrangianMinProblem")
        self.original_problem = original_problem
        self.fixed_multipliers_inequality = [0.0] * len(self.original_problem.evaluate(self.original_problem.bounds[0]).constraints_inequality or [])
        self.fixed_multipliers_equality = [0.0] * len(self.original_problem.evaluate(self.original_problem.bounds[0]).constraints_equality or [])

    def set_fixed_multipliers(self, inequality_multipliers, equality_multipliers):
        """Updates the fixed Lagrangian multipliers for the next generation.

        This method is called by the main `CoevolutionaryLagrangianSolver` before
        the objective solver performs its next step.

        Args:
            inequality_multipliers (List[float]): The new set of fixed mu* values.
            equality_multipliers (List[float]): The new set of fixed lambda* values.
        """
        self.fixed_multipliers_inequality = inequality_multipliers
        self.fixed_multipliers_equality = equality_multipliers

    def evaluate(self, solution: list[float]) -> Evaluation:
        """Calculates the Lagrangian value L(x, mu*, lambda*).

        This evaluation treats the problem as unconstrained, returning only a
        single fitness value representing the Lagrangian.

        Args:
            solution (SolutionType): The candidate solution 'x' to evaluate.

        Returns:
            Evaluation[float]: An Evaluation object where `fitness` is the
                Lagrangian value. The constraint fields are empty.
        """
        # Evaluate the original problem to get f(x), g(x), and h(x)
        original_eval = self.original_problem.evaluate(solution)
        fx = original_eval.fitness
        gx = original_eval.constraints_inequality or []
        hx = original_eval.constraints_equality or []

        # Calculate L(x, mu*, lambda*)
        # This is based on Definition 2.5 from your document
        lagrangian_value = fx
        lagrangian_value += sum(s * g for s, g in zip(self.fixed_multipliers_inequality, gx))
        lagrangian_value += sum(l * h for l, h in zip(self.fixed_multipliers_equality, hx))

        # This problem is now unconstrained from the solver's perspective
        return Evaluation(fitness=lagrangian_value)

    def is_dynamic(self) -> tuple[bool, bool]:
        """Delegates the check for dynamic properties to the original problem.

        Returns:
            Tuple[bool, bool]: A tuple indicating if the original problem's
                objectives or constraints are dynamic.
        """
        return self.original_problem.is_dynamic()


class _LagrangianMaxProblem(Problem):
    """An internal proxy problem for the multiplier-space solver ('max' swarm).

    This class wraps the original problem to create the search space for the
    Lagrangian multipliers (mu and lambda). Its fitness function is L(x*, mu, lambda), where
    the solution 'x*' is fixed for the current generation. Since library solvers
    typically minimize, this class returns -L(x*, mu, lambda) to achieve maximization.

    Attributes:
        original_problem (Problem): A reference to the user-defined constrained
            problem.
        fixed_solution_eval (Evaluation): The evaluation result of the best
            solution (x*) from the objective swarm.
        num_inequality (int): The number of inequality constraints.
    """
    def __init__(self,
                 original_problem: Problem[SolutionType, float],
                 fixed_solution: SolutionType):
        """Initializes the proxy problem for the multiplier space.

        Args:
            original_problem (Problem): The original constrained problem.
            fixed_solution (SolutionType): An initial solution 'x' used to
                determine the number of constraints and thus the dimension
                of the multiplier search space.
        """
        num_inequality = len(original_problem.evaluate(fixed_solution).constraints_inequality or [])
        num_equality = len(original_problem.evaluate(fixed_solution).constraints_equality or [])
        dimension = num_inequality + num_equality

        # Multipliers for inequality constraints (mu) must be >= 0
        # Multipliers for equality constraints (lambda) are unrestricted
        lower_bounds = [0.0] * num_inequality + [-float('inf')] * num_equality
        upper_bounds = [float('inf')] * (num_inequality + num_equality)

        super().__init__(dimension, (lower_bounds, upper_bounds), "LagrangianMaxProblem")
        self.original_problem = original_problem
        self.fixed_solution_eval = original_problem.evaluate(fixed_solution)
        self.num_inequality = num_inequality

    def set_fixed_solution(self, solution):
        """Updates the fixed solution 'x*' for the next generation.

        This method is called by the main `CoevolutionaryLagrangianSolver` before
        the multiplier solver performs its next step.

        Args:
            solution (SolutionType): The new fixed solution 'x*'.
        """
        self.fixed_solution_eval = self.original_problem.evaluate(solution)

    def evaluate(self, solution: list[float]) -> Evaluation:
        """Calculates -L(x*, mu, lambda) for maximization.

        The `solution` argument here is a vector of concatenated Lagrangian
        multipliers [mu_1, ..., mu_n, lambda_1, ..., lambda_m].

        Args:
            solution (List[float]): A candidate vector of multipliers.

        Returns:
            Evaluation[float]: An Evaluation object where `fitness` is the
                negated Lagrangian value.
        """
        # Unpack multipliers
        inequality_multipliers = solution[:self.num_inequality]
        equality_multipliers = solution[self.num_inequality:]

        fx = self.fixed_solution_eval.fitness
        gx = self.fixed_solution_eval.constraints_inequality or []
        hx = self.fixed_solution_eval.constraints_equality or []

        # Calculate L(x*, mu, lambda)
        lagrangian_value = fx
        lagrangian_value += sum(s * g for s, g in zip(inequality_multipliers, gx))
        lagrangian_value += sum(l * h for l, h in zip(equality_multipliers, hx))

        # Return the negative value because we want to MAXIMIZE L
        return Evaluation(fitness=-lagrangian_value)

    def is_dynamic(self) -> tuple[bool, bool]:
        """Delegates the check for dynamic properties to the original problem.

        Returns:
            Tuple[bool, bool]: A tuple indicating if the original problem's
                objectives or constraints are dynamic.
        """
        return self.original_problem.is_dynamic()


class CoevolutionaryLagrangianSolver(Solver):
    """A meta-solver for constrained optimization using a co-evolutionary
    Lagrangian framework.

    This solver transforms a constrained problem into an unconstrained min-max
    problem, which it solves using two cooperating ("co-evolving") populations
    of standard solvers. It is generic and can be configured with any two
    solver classes from the `cilpy` library.

    Attributes:
        objective_solver (Solver): The subordinate solver instance that searches
            the solution space of the original problem.
        multiplier_solver (Solver): The subordinate solver instance that searches
            the space of the Lagrangian multipliers.
        min_problem (_LagrangianMinProblem): The internal proxy problem for the
            objective solver.
        max_problem (_LagrangianMaxProblem): The internal proxy problem for the
            multiplier solver.
    """

    def __init__(self,
                 name: str,
                 problem: Problem,
                 objective_solver_class,
                 multiplier_solver_class,
                 objective_solver_params: dict,
                 multiplier_solver_params: dict):
        """Initializes the CoevolutionaryLagrangianSolver.

        Args:
            problem (Problem): The constrained optimization problem to solve.
            name (str): The name of this solver instance.
            objective_solver_class (Type[Solver]): The class of the solver to use
                for the objective space (e.g., `GA`, `PSO`).
            multiplier_solver_class (Type[Solver]): The class of the solver to use
                for the multiplier space.
            objective_solver_params (Dict[str, Any]): A dictionary of parameters
                to initialize the objective solver.
            multiplier_solver_params (Dict[str, Any]): A dictionary of parameters
                to initialize the multiplier solver.
        """

        super().__init__(problem, name=name)

        # 1. Create the proxy problems
        # We need an initial solution to dimension the multiplier problem
        initial_solution = [problem.bounds[0][i] for i in range(problem.dimension)]
        self.min_problem = _LagrangianMinProblem(problem)
        self.max_problem = _LagrangianMaxProblem(problem, initial_solution)

        # 2. Instantiate the internal solvers
        self.objective_solver = objective_solver_class(
            problem=self.min_problem,
            **objective_solver_params
        )
        self.multiplier_solver = multiplier_solver_class(
            problem=self.max_problem,
            **multiplier_solver_params
        )

    def step(self) -> None:
        """Performs one co-evolutionary step.

        This process involves:
        1. Getting the best individuals from each population.
        2. Updating the fitness landscape of each sub-problem using the best
           individual from the other population.
        3. Advancing each subordinate solver by one iteration.
        """

        # 1. Get the best individuals from each population
        best_solution, _ = self.objective_solver.get_result()[0]
        best_multipliers, _ = self.multiplier_solver.get_result()[0]
        
        num_inequality = self.max_problem.num_inequality
        inequality_multipliers = best_multipliers[:num_inequality]
        equality_multipliers = best_multipliers[num_inequality:]

        # 2. Update the fitness landscapes for the sub-solvers
        # The 'min' problem gets the best multipliers from the 'max' solver
        self.min_problem.set_fixed_multipliers(inequality_multipliers, equality_multipliers)
        # The 'max' problem gets the best solution from the 'min' solver
        self.max_problem.set_fixed_solution(best_solution)

        # 3. Perform one step of each sub-solver
        self.objective_solver.step()
        self.multiplier_solver.step()

        # Optional: Handle dynamic changes
        is_obj_dyn, is_con_dyn = self.problem.is_dynamic()
        if is_obj_dyn or is_con_dyn:
            # TODO: A hook for handling dynamic problems can be added here.
            # For DCOPs, if the number of constraints changes, the dimension of
            # self.max_problem changes, and self.multiplier_solver would need to
            # be re-initialized or adapted.
            pass

    def get_result(self) -> list[tuple[list[float], Evaluation]]:
        """Returns the best solution found in the objective space.

        The returned solution is evaluated against the *original* constrained
        problem, providing the user with the true fitness and constraint
        violation information.

        Returns:
            List[Tuple[SolutionType, Evaluation[float]]]: A list containing a
                single tuple of (best_solution, evaluation_on_original_problem).
        """
        best_solution, _ = self.objective_solver.get_result()[0]
        final_evaluation = self.problem.evaluate(best_solution)
        return [(best_solution, final_evaluation)]

```

This is the experiment runner which orchestrates interaction between components:
```
# cilpy/runner.py
import time
import csv
from typing import Any, Dict, List, Type, Sequence

from cilpy.problem import Problem
from cilpy.solver import Solver


class ExperimentRunner:
    """
    A generic runner for conducting computational intelligence experiments.

    This class orchestrates the process of running multiple optimization
    algorithms (solvers) on a collection of problems for a specified number of
    independent runs and iterations. It handles the setup, execution, and result
    logging for each experiment.
    """

    def __init__(self,
                 problems: Sequence[Problem],
                 solver_configurations: List[Dict[str, Any]],
                 num_runs: int,
                 max_iterations: int):
        """
        Initializes the ExperimentRunner.

        Args:
            problems (List[Problem]): A list of problem instances to be solved.
                Each problem must implement the `Problem` interface.
            solver_configurations (List[Dict[str, Any]]): A list of solver
                configurations. Each configuration is a dictionary that should
                contain:
                - "class" (Type[Solver]): The solver class (e.g., `PSO`, `GA`).
                - "params" (Dict): A dictionary of parameters for the solver.
                  The 'problem' parameter will be set by the runner.
            num_runs (int): The number of independent runs to perform for each
                solver-problem pair.
            max_iterations (int): The number of iterations to run each solver
                for in a single run.
        """
        self.problems = problems
        self.solver_configurations = solver_configurations
        self.num_runs = num_runs
        self.max_iterations = max_iterations

    def run_experiments(self):
        """
        Executes the full suite of experiments defined during initialization.

        This method iterates through each problem and applies every configured
        solver to it, performing the specified number of runs and iterations.
        Results are logged to separate CSV files for each problem-solver pair.
        """
        total_start_time = time.time()
        print("======== Starting All Experiments ========")

        for problem in self.problems:
            print(f"\n--- Processing Problem: {problem.name} ---")
            for config in self.solver_configurations:
                solver_class = config["class"]
                # Use .copy() to avoid modifying the original params dict
                solver_params = config["params"].copy()

                # Add the current problem to the solver's parameters
                current_solver_params = solver_params
                current_solver_params["problem"] = problem

                solver_name = current_solver_params.get("name")
                output_file_path = f"{problem.name}_{solver_name}.out.csv"

                print(f"\n  -> Starting Experiment: {solver_name} on {problem.name}")
                print(f"     Configuration: {self.num_runs} runs, {self.max_iterations} iterations/run.")
                print(f"     Results will be saved to: {output_file_path}")

                self._run_single_experiment(solver_class, current_solver_params, output_file_path)

        total_end_time = time.time()
        print("\n======== All Experiments Finished ========")
        print(f"Total execution time: {total_end_time - total_start_time:.2f}s")

    def _run_single_experiment(
            self,
            solver_class: Type[Solver],
            solver_params: Dict,
            output_file: str):
        """
        Runs and logs a single experiment for a given solver and problem.

        Args:
            solver_class (Type[Solver]): The class of the solver to be used.
            solver_params (Dict): The parameters for initializing the solver.
            output_file (str): The path to the CSV file where results will be
                saved.
        """
        header = ["run", "iteration", "result"]
        experiment_start_time = time.time()

        with open(output_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for run_id in range(1, self.num_runs + 1):
                run_start_time = time.time()
                print(f"     --- Starting Run {run_id}/{self.num_runs} ---")

                # Re-instantiate the solver for each run to ensure independence
                solver = solver_class(**solver_params)

                for iteration in range(1, self.max_iterations + 1):
                    solver.step()
                    result = solver.get_result()

                    # Log data for this iteration
                    writer.writerow([run_id, iteration, result])

                run_end_time = time.time()
                final_result = solver.get_result()
                print(
                    f"     Run {run_id} finished in {run_end_time - run_start_time:.2f}s. "
                    f"Best result: {final_result}"
                )

        experiment_end_time = time.time()
        solver_name = solver_params.get('name', solver_class.__name__)
        problem_name = solver_params['problem'].name
        print(f"  -> Experiment for {solver_name} on {problem_name} "
              f"finished in {experiment_end_time - experiment_start_time:.2f}s.")


if __name__ == '__main__':
    from cilpy.problem.functions import Sphere, Ackley
    from cilpy.solver.pso import PSO
    from cilpy.solver.ga import GA

    # --- 1. Define the Problems ---
    dim = 3
    dom = (-5.12, 5.12)
    problems_to_run = [
        Sphere(dimension=dim, domain=dom),
        Ackley(dimension=dim, domain=dom)
    ]

    # --- 2. Define the Solver Configurations ---
    solver_configs = [
        {
            "class": GA,
            "params": {
                "name": "GA_HighMutation",
                "population_size": 30,
                "crossover_rate": 0.2,
                "mutation_rate": 0.3,
                "tournament_size": 7,
            }
        },
        {
            "class": PSO,
            "params": {
                "name": "PSO_Standard",
                "swarm_size": 30,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
                "k": 1,
            }
        }
    ]

    # --- 3. Define the Experiment parameters ---
    number_of_runs = 5
    max_iter = 1000

    # --- 4. Create and run the experiments ---
    runner = ExperimentRunner(
        problems=problems_to_run,
        solver_configurations=solver_configs,
        num_runs=number_of_runs,
        max_iterations=max_iter
    )
    runner.run_experiments()

```

This is an example of an experiment:
```
# examples/ga_exa.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.functions import Sphere, Ackley
from cilpy.solver.ga import GA

# --- 1. Define the Problems ---
dim = 3
dom = (-5.12, 5.12)
problems_to_run = [
    Sphere(dimension=dim, domain=dom),
    Ackley(dimension=dim, domain=dom)
]

# --- 2. Define the Solvers and their parameters ---
# Note: The 'problem' parameter is omitted here as the runner will assign it.
solver_configs = [
    {
        "class": GA,
        "params": {
            "name": "GA_HighMutation",
            "population_size": 30,
            "crossover_rate": 0.2,
            "mutation_rate": 0.3, # Higher mutation
            "tournament_size": 7,
        }
    },
    {
        "class": GA,
        "params": {
            "name": "GA_LowMutation",
            "population_size": 30,
            "crossover_rate": 0.2,
            "mutation_rate": 0.1, # Lower mutation
            "tournament_size": 7,
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 1
max_iter = 10

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
```

And this is an example of how the cooperative co-evolutionary Lagrangian solver
is used:
```
# examples/ga_ri_cc_exa.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark, generate_mpb_configs
from cilpy.solver.ga import RIGA
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver

# --- 1. Define the Problems ---
all_mpb_configs = generate_mpb_configs(dimension=5)
f_params = all_mpb_configs['A3L']
g_params = all_mpb_configs['P1R']

problems_to_run = [
    ConstrainedMovingPeaksBenchmark(
        f_params=f_params,
        g_params=g_params,
        name="CMPB_A3L_P1R"
    )
]

# --- 2. Define the Solvers and their parameters ---
solver_configs = [
    {
        # This is the co-evolutionary solver configuration
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCLS_with_RIGAs",
            "objective_solver_class": RIGA,
            "multiplier_solver_class": RIGA,
            "objective_solver_params": {
                "name": "ObjectiveGA",
                "population_size": 40,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "immigrant_rate": 0.2,
                "tournament_size": 3,
            },
            "multiplier_solver_params": {
                "name": "MultiplierGA",
                "population_size": 40,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "immigrant_rate": 0.2,
                "tournament_size": 3,
            }
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 1
max_iter = 5

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()

```

Write the overview documentation for cilpy, write the docs/index.md file.

I want the documentation I write to be in one place, so preferably in the code.