"""
Implementation following:
Jun Sun, Bin Feng and Wenbo Xu, "Particle swarm optimization with particles
having quantum behavior," Proceedings of the 2004 Congress on Evolutionary
Computation (IEEE Cat. No.04TH8753), Portland, OR, USA, 2004, pp. 325-331 Vol.1,
doi: 10.1109/CEC.2004.1330875. keywords: {Particle swarm optimization;Potential
well;Organisms;Equations;Sun;Information technology;Information analysis;
Evolutionary computation;Educational institutions;Birds},
"""
import random
import math

def objective_func(position: list[float], t: int) -> float:
    """
    The Sphere function with minimum drifts along the vector
    v(t) = (sin(0.05 t), sin(0.05 t), …).

    f(x) = sum((x_i - v_i(t))^2) for i=1 to n
    """
    drift = math.sin(0.05 * t)
    return sum((x - drift) ** 2 for x in position)

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
    def __init__(self, dim, min_x, max_x):
        self.pos = [random.uniform(min_x, max_x) for _ in range(dim)]
        self.pbest_pos = list(self.pos)
        self.pbest_fitness = float('inf')

    def update_pos(self, mbest_pos: list[float], gbest_pos: list[float],
                   alpha: float, min_x: float, max_x: float,
                   distribution_strategy: callable) -> None:
        for i in range(len(self.pos)):
            # Calculate the local attractor for each dimension
            phi = random.random()
            local_attractor = phi * self.pbest_pos[i] + (1 - phi) * gbest_pos[i]

            # Calculate the new position
            self.pos[i] = distribution_strategy(local_attractor, self.pos[i], mbest_pos[i], alpha)

            # Apply bounds to position
            if self.pos[i] < min_x:
                self.pos[i] = min_x
            elif self.pos[i] > max_x:
                self.pos[i] = max_x

def qpso(dim: int,
         min_x: float,
         max_x: float,
         objective_func: callable,
         distribution_strategy: callable,
         n: int = 30,
         iterations: int = 100,
         alpha_start: float = 1.0,
         alpha_end: float = 0.5) -> list[float]:
    
    # Create and initialize an dim-dimensional swarm
    swarm = [Particle(dim, min_x, max_x) for _ in range(n)]
    global_best_pos = None
    global_best_fitness = float('inf')

    # Initial evaluation to set personal bests and initialize global best
    for particle in swarm:
        current_fitness = objective_func(particle.pos, 0)
        particle.pbest_fitness = current_fitness
        particle.pbest_pos = list(particle.pos)

        if current_fitness < global_best_fitness:
            global_best_fitness = current_fitness
            global_best_pos = list(particle.pos)

    for iteration in range(iterations):
        # Calculate the mean best position
        mbest_pos = [sum(p.pbest_pos[i] for p in swarm) / n for i in range(dim)]

        # Linearly decreasing contraction-expansion coefficient
        alpha = alpha_start - (iteration / iterations) * (alpha_start - alpha_end)

        # Update personal and global bests
        for particle in swarm:
            current_fitness = objective_func(particle.pos, iteration)
            if current_fitness < particle.pbest_fitness:
                particle.pbest_fitness = current_fitness
                particle.pbest_pos = list(particle.pos)
            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_pos = list(particle.pos)

        # Update particle positions
        for particle in swarm:
            particle.update_pos(mbest_pos, global_best_pos, alpha, min_x, max_x, distribution_strategy)

        # Optional: Print progress
        if iteration % (iterations // 10) == 0 or iteration == iterations - 1:
            print(f"Iteration {iteration+1}/{iterations}: Global Best Fitness = {global_best_fitness:.6f}")

    return global_best_pos

if __name__ == "__main__":
    dim = 2
    min_x = -10
    max_x = 10
    iterations = 1000

    best_position = qpso(dim, min_x, max_x, objective_func, distribution_strategy, iterations)

    print("\n--- Optimization Complete ---")
    print(f"Best Position Found: {best_position}")
    print(f"Objective Function Value: {objective_func(best_position, iterations):.6f}")