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

    def update_vel(self, local_best_pos: list[float], w: float, c1: float, c2: float) -> None:
        for i in range(len(self.pos)):
            r1 = random.random()
            r2 = random.random()

            cognitive_component = c1 * r1 * (self.pbest_pos[i] - self.pos[i])
            social_component = c2 * r2 * (local_best_pos[i] - self.pos[i])
            self.vel[i] = w * self.vel[i] + cognitive_component + social_component

    def update_pos(self, min_x: float, max_x: float) -> None:
        for i in range(len(self.pos)):
            self.pos[i] += self.vel[i]
            # Apply bounds to position
            if self.pos[i] < min_x:
                self.pos[i] = min_x
            elif self.pos[i] > max_x:
                self.pos[i] = max_x

def lbest_pso(dim: int,
              min_x: float,
              max_x: float,
              objective_func: callable,
              n: int = 30,
              iterations: int = 100,
              neighbourhood_size: int = 5,
              w_start: float = 0.9,
              w_end: float = 0.4,
              c1: float = 2.0,
              c2: float = 2.0) -> list[float]:

    # Create and initialize an dim-dimensional swarm
    swarm = [Particle(dim, min_x, max_x) for _ in range(n)]
    global_best_pos = None
    global_best_fitness = float('inf')

    # Initial evaluation to set personal bests and initialize global best
    for particle in swarm:
        current_fitness = objective_func(particle.pos)
        particle.pbest_fitness = current_fitness
        particle.pbest_pos = list(particle.pos)

        if current_fitness < global_best_fitness:
            global_best_fitness = current_fitness
            global_best_pos = list(particle.pos)
    
    print(f"Initial Global Best Fitness: {global_best_fitness:.6f}")

    for iteration in range(iterations):
        w = w_start - (iteration / iterations) * (w_start - w_end)

        # Update all personal and local bests for the current iteration
        for i, particle in enumerate(swarm):
            current_fitness = objective_func(particle.pos)

            # Update personal best position
            if current_fitness < particle.pbest_fitness:
                particle.pbest_fitness = current_fitness
                particle.pbest_pos = list(particle.pos)

            # Update overall global best (for reporting purposes)
            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_pos = list(particle.pos)

        # Update velocities and positions of all particles using lbest
        for i, particle in enumerate(swarm):
            # Determine the neighbourhood for the current particle (ring topology)
            lbest_idx = i
            lbest_fitness = swarm[i].pbest_fitness

            # Check neighbours
            for j in range(1, (neighbourhood_size // 2) + 1):
                # Left neighbour
                left_neighbour_idx = (i - j + n) % n
                if swarm[left_neighbour_idx].pbest_fitness < lbest_fitness:
                    lbest_fitness = swarm[left_neighbour_idx].pbest_fitness
                    lbest_idx = left_neighbour_idx
                
                # Right neighbour
                right_neighbour_idx = (i + j) % n
                if swarm[right_neighbour_idx].pbest_fitness < lbest_fitness:
                    lbest_fitness = swarm[right_neighbour_idx].pbest_fitness
                    lbest_idx = right_neighbour_idx
            
            lbest_pos = swarm[lbest_idx].pbest_pos

            # Update velocity and position using the local best
            particle.update_vel(lbest_pos, w, c1, c2)
            particle.update_pos(min_x, max_x)

        # Optional: Print progress
        if iteration % (iterations // 10) == 0 or iteration == iterations - 1:
            print(f"Iteration {iteration+1}/{iterations}: Global Best Fitness = {global_best_fitness:.6f}")

    return global_best_pos

if __name__ == "__main__":
    dim = 2
    min_x = -10.0
    max_x = 10.0
    
    best_position = lbest_pso(dim, min_x, max_x, objective_func)

    print("\n--- Optimization Complete ---")
    print(f"Best Position Found: {best_position}")
    print(f"Objective Function Value at Best Position: {objective_func(best_position):.6f}")
