from typing import Tuple

import random

def objective_func(position: list[float]) -> float:
    """
    The Sphere function (objective function).
    f(x) = sum(x_i^2) for i=1 to n
    The global minimum is at x = (0, ..., 0) with f(x) = 0.
    """
    return sum(x**2 for x in position)

class Particle:
    def __init__(self, dim, min_x, max_x):
        self.pos = [random.uniform(min_x, max_x) for _ in range(dim)]
        self.vel = [0.0 for _ in range(dim)]
        self.pbest_pos = list(self.pos)
        self.pbest_fitness = float('inf')

    def update_vel(self,
                   gbest_pos: list[float],
                   w: float,
                   c1: float,
                   c2: float) -> None:
        for i in range(len(self.pos)):
            r1 = random.random()
            r2 = random.random()

            cognitive_component = c1*r1*(self.pbest_pos[i] - self.pos[i])
            social_component = c2*r2*(gbest_pos[i] - self.pos[i])
            self.vel[i] = w*self.vel[i] + cognitive_component + social_component

    def update_pos(self, min_x: float, max_x: float) -> None:
        for i in range(len(self.pos)):
            self.pos[i] += self.vel[i]
            # Apply bounds to position
            self.pos[i] = max(min_x, min(self.pos[i], max_x))

def gbest_pso(dim: int,
              min_x: float,
              max_x: float,
              objective_func: callable,
              n: int=30,
              iterations: int=100,
              w_start: float=0.9,
              w_end: float=0.4,
              c1: float=2.0,
              c2: float=2.0) -> Tuple[list[float], float]:
    
    # Create and initialize an dim-dimensional swarm
    swarm = [Particle(dim, min_x, max_x) for _ in range(n)]
    best_solution = None
    best_fitness = float('inf')

    # Initial evaluation to set personal and global bests
    for particle in swarm:
        current_fitness = objective_func(particle.pos)
        particle.pbest_fitness = current_fitness
        particle.pbest_pos = list(particle.pos)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = list(particle.pos)
    
    for iteration in range(iterations):
        # Linearly decreasing inertia weight
        w = w_start - (iteration / iterations)*(w_start - w_end)

        # Update all personal and global bests for the current iteration
        for particle in swarm:
            current_fitness = objective_func(particle.pos)

            # Update personal best position
            if current_fitness < particle.pbest_fitness:
                particle.pbest_fitness = current_fitness
                particle.pbest_pos = list(particle.pos)

            # Update global best position
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = list(particle.pos)
        
        # Update velocities and positions of all particles
        for particle in swarm:
            particle.update_vel(best_solution, w, c1, c2)
            particle.update_pos(min_x, max_x)

        # Optional: Print progress
        if iteration % (iterations // 10) == 0 or iteration == iterations - 1:
            iter = (iteration+1)/iterations
            print(f"Iteration {iter}: Best Fitness = {best_fitness:.6f}")

    return best_solution, best_fitness
        
if __name__ == "__main__":
    dim = 2
    min_x = -10.0
    max_x = 10.0

    solution, fitness_value = gbest_pso(dim, min_x, max_x, objective_func)
    
    print("\n--- Results ---")
    print(f"Best solution found: {solution}")
    print(f"Fitness of the best solution: {fitness_value}")
