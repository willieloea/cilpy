import math
import random

# Dynamic objective function
def objective_func_d(position: list[float], t: int) -> float:
    """
    Dynamic Sphere function with drifts along the vector
    v(t) = (sin(0.05*t), sin(0.05*t), ...).

    f(x) = sum((x_i - v_i(t))^2) for i=1 to n
    """

    drift = math.sin(0.05 * t)
    return sum((x-drift)**2 for x in position)

class Particle:
    def __init__(self, dim: int, min_x: float, max_x: float):
        self.pos = [random.uniform(min_x, max_x) for _ in range(dim)]
        self.pbest_pos = list(self.pos)
        self.pbest_fitness = float('inf')
    
    def update_pos(self, mbest_pos: list[float], gbest_pos: list[float],
                   alpha: float, min_x: float, max_x: float) -> None:
        for i in range(len(self.pos)):
            # Calculate local attractor for each dimension
            phi = random.random()
            p = (phi * self.pbest_pos[i] + (1 - phi) * gbest_pos[i])

            u = random.random()

            if random.random() < 0.5:
                self.pos[i] = p + alpha * abs(mbest_pos[i] - self.pos[i]) * math.log(1 / u)
            else:
                self.pos[i] = p - alpha * abs(mbest_pos[i] - self.pos[i]) * math.log(1 / u)
            
            # Apply bouds to position
            if self.pos[i] < min_x:
                self.pos[i] = min_x
            elif self.pos[i] > max_x:
                self.pos[i] = max_x

def qpso(dim: int,
         min_x: float, 
         max_x: float, 
         objective_func: callable,
         n: int=30,
         iterations: int=100,
         alpha_start: float=1.0,
         alpha_end: float=0.5) -> list[float]:

    gbest_pos = None

    # Create and initialize an dim-dimensional swarm
    swarm = [Particle(dim, min_x, max_x) for _ in range(n)]

    gbest_pos = None
    gbest_fitness = float('inf')

    # Initial evaluation to set personal and global bests
    for particle in swarm:
        current_fitness = objective_func_d(particle.pos, 0)
        particle.pbest_fitness = current_fitness
        particle.pbest_pos = list(particle.pos)

        if current_fitness < gbest_fitness:
            gbest_fitness = current_fitness
            gbest_pos = list(particle.pos)

    # Perform QPSO
    for iteration in range(iterations):
        # Calculate the mand best position (mbest)
        mbest_pos = [0.0] * dim
        for particle in swarm:
            for i in range(dim):
                mbest_pos[i] += particle.pbest_pos[i]
        mbest_pos = [x / n for x in mbest_pos]

        # Linearly decreasing contraction-expansion coefficient
        alpha = alpha_start - (iteration / iterations) * (alpha_start - alpha_end)

        # Update personal and global bests
        for particle in swarm:
            current_fitness = objective_func_d(particle.pos, iteration)

            # Update personal best position
            if current_fitness < particle.pbest_fitness:
                particle.pbest_fitness = current_fitness
                particle.pbest_pos = list(particle.pos)

            # Update global best position
            if current_fitness < gbest_fitness:
                gbest_fitness = current_fitness
                gbest_pos = list(particle.pos)
        
        # Update particle positions
        for particle in swarm:
            particle.update_pos(mbest_pos, gbest_pos, alpha, min_x, max_x)

        # Optional: Print progress
        if iteration % (iterations // 10) == 0 or iteration == iterations - 1:
            print(f"Iteration {iteration+1}/{iterations}: Global Best Fitness = {gbest_fitness:.6f}")
    
    # Return best solution
    return gbest_pos

if __name__ == "__main__":
    dim = 2
    min_x = -10.0
    max_x = 10.0

    best_position = qpso(dim, min_x, max_x, objective_func_d)

    print("\n--- Optimization Complete ---")
    print(f"Best Position Found: {best_position}")
    print(f"Objective Function Value at Best Position: {objective_func_d(best_position, 100):.6f}")
