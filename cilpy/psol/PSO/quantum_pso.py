from typing import Tuple, Callable

import random
import math

def objective_func(position: list[float], t: int) -> float:
    """
    The Sphere function (objective function).
    f(x) = sum(x_i^2) for i=1 to n
    The global minimum is at x = (0, ..., 0) with f(x) = 0.
    """
    return sum(x**2 for x in position)

def uniform_distribution(local_attractor: float, current_pos: float,
                         mbest_pos: float, alpha: float) -> float:
    """
    Position update strategy based on a Uniform distribution.

    This strategy defines a search window centered around the local attractor
    and selects a new position with equal probability from anywhere within
    that window.
    """
    # Define the radius of the uniform search window.
    char_length = alpha * abs(mbest_pos - current_pos)
    
    lower_bound = local_attractor - char_length
    upper_bound = local_attractor + char_length
    
    return random.uniform(lower_bound, upper_bound)

def guassian_distribution(local_attractor: float, current_pos: float,
                          mbest_pos: float, alpha: float) -> float:
    """
    Position update strategy based on a Gaussian distribution.

    This strategy selects a new position from a Gaussian (normal) distribution
    centered at the local attractor. The standard deviation of the distribution
    is determined by the characteristic length, which is proportional to the
    distance between the mean best position and the current position.
    """
    # Define the standard deviation of the Gaussian distribution.
    char_length = alpha * abs(mbest_pos - current_pos)
    
    return random.gauss(mu=local_attractor, sigma=char_length)

def distribution_strategy(local_attractor: float, current_pos: float,
                          mbest_pos: float, alpha: float) -> float:
    return uniform_distribution(local_attractor, current_pos, mbest_pos, alpha)

class Particle:
    def __init__(self, dim: int, min_x: float, max_x: float):
        self.pos = [random.uniform(min_x, max_x) for _ in range(dim)]
        self.pbest_pos = list(self.pos)
        self.pbest_fitness = float('inf')

    def update_pos(self, mbest_pos: list[float], gbest_pos: list[float],
                   alpha: float, min_x: float, max_x: float,
                   distribution_strategy: Callable[[float, float, float, float],
                                                   float]
                   ) -> None:
        for i in range(len(self.pos)):
            # Calculate the local attractor for each dimension
            phi = random.random()
            local_attractor = phi * self.pbest_pos[i] + (1 - phi) * gbest_pos[i]

            # Calculate the new position
            self.pos[i] = distribution_strategy(local_attractor, self.pos[i], 
                                                mbest_pos[i], alpha)

            # Apply bounds to position
            self.pos[i] = max(min_x, min(self.pos[i], max_x))

def qpso(dim: int,
         min_x: float,
         max_x: float,
         objective_func: Callable[[list[float], int], float],
         distribution_strategy: Callable[[float, float, float, float], float],
         n: int = 30,
         iterations: int = 1000,
         alpha_start: float = 1.0,
         alpha_end: float = 0.5) -> Tuple[list[float], float]:
    
    # Create and initialize an dim-dimensional swarm
    swarm = [Particle(dim, min_x, max_x) for _ in range(n)]
    best_solution: list[float] = []
    best_fitness = float('inf')

    # Initial evaluation to set personal bests and initialize global best
    for particle in swarm:
        current_fitness = objective_func(particle.pos, 0)
        particle.pbest_fitness = current_fitness
        particle.pbest_pos = list(particle.pos)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = list(particle.pos)

    for iteration in range(iterations):
        # Calculate the mean best position
        mbest_pos = [sum(p.pbest_pos[i] for p in swarm) / n for i in range(dim)]

        # Linearly decreasing contraction-expansion coefficient
        alpha = alpha_start 
        - (iteration / iterations) * (alpha_start - alpha_end)

        # Update personal and global bests
        for particle in swarm:
            current_fitness = objective_func(particle.pos, iteration)
            if current_fitness < particle.pbest_fitness:
                particle.pbest_fitness = current_fitness
                particle.pbest_pos = list(particle.pos)
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = list(particle.pos)

        # Update particle positions
        for particle in swarm:
            particle.update_pos(mbest_pos, best_solution, alpha, min_x, max_x,
                                distribution_strategy)

        # Optional: Print progress
        if iteration % (iterations // 10) == 0 or iteration == iterations - 1:
            iter = (iteration+1)/iterations
            print(f"Iteration {iter}: Best Fitness = {best_fitness:.6f}")

    return best_solution, best_fitness

if __name__ == "__main__":
    dim = 2
    min_x = -10
    max_x = 10

    solution, fitness_value = qpso(dim, min_x, max_x, objective_func,
                                   distribution_strategy)
    
    print("\n--- Results ---")
    print(f"Best solution found: {solution}")
    print(f"Fitness of the best solution: {fitness_value}")
