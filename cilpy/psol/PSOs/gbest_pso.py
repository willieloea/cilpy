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

    def update_vel(self, gbest_pos: list[float], c1: float, c2: float) -> None:
        for i in range(len(self.pos)):
            r1 = random.random()
            r2 = random.random()

            cognitive_component = c1 * r1 * (self.pbest_pos[i] - self.pos[i])
            social_component = c2 * r2 * (gbest_pos[i] - self.pos[i])
            self.vel[i] = self.vel[i] + cognitive_component + social_component

    def update_pos(self, min_x: float, max_x: float) -> None:
        for i in range(len(self.pos)):
            self.pos[i] += self.vel[i]
            # Apply bounds to position
            if self.pos[i] < min_x:
                self.pos[i] = min_x
            elif self.pos[i] > max_x:
                self.pos[i] = max_x

def gbest_pso(dim: int,
              min_x: float,
              max_x: float,
              objective_func: callable,
              n: int=30,
              iterations: int=100,
              c1: float=2.0,
              c2: float=2.0) -> list[float]:

    # Create and initialize an dim-dimensional swarm
    swarm = [Particle(dim, min_x, max_x) for _ in range(n)]

    # Initialize gbest_pos and gbest_fitness
    gbest_pos = None
    gbest_fitness = float('inf')

    # Initial evaluation to set personal and global bests
    for particle in swarm:
        current_fitness = objective_func(particle.pos)
        particle.pbest_fitness = current_fitness
        particle.pbest_pos = list(particle.pos) # Ensure it's a copy

        if current_fitness < gbest_fitness:
            gbest_fitness = current_fitness
            gbest_pos = list(particle.pos) # Ensure it's a copy

    for iteration in range(iterations):
        # Update all personal and global bests for the current iteration
        for particle in swarm:
            current_fitness = objective_func(particle.pos)

            # Update personal best position
            if current_fitness < particle.pbest_fitness:
                particle.pbest_fitness = current_fitness
                particle.pbest_pos = list(particle.pos)

            # Update global best position
            if current_fitness < gbest_fitness:
                gbest_fitness = current_fitness
                gbest_pos = list(particle.pos)
        
        # Update velocities and positions of all particles
        for particle in swarm:
            particle.update_vel(gbest_pos, c1, c2)
            particle.update_pos(min_x, max_x)

        # Optional: Print progress
        if iteration % (iterations // 10) == 0 or iteration == iterations - 1:
            print(f"Iteration {iteration+1}/{iterations}: Global Best Fitness = {gbest_fitness:.6f}")

    return gbest_pos
        
if __name__ == "__main__":
    dim = 2
    min_x = -10
    max_x = 10

    best_position = gbest_pso(dim, min_x, max_x, objective_func)

    print("\n--- Optimization Complete ---")
    print(f"Best Position Found: {best_position}")
    print(f"Objective Function Value at Best Position: {objective_func(best_position):.6f}")
