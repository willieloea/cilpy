from typing import Tuple

import random

def objective_func(position: list[float]) -> float:
    """
    The Sphere function (objective function).
    f(x) = sum(x_i^2) for i=1 to n
    The global minimum is at x = (0, ..., 0) with f(x) = 0.
    """
    return sum(x**2 for x in position)

def de_rand_1_bin(dim: int,
                  min_x: float,
                  max_x: float,
                  objective_func: callable,
                  n: int = 100,
                  max_generations: int = 500,
                  scale_factor: float = 0.5,
                  crossover_prob: float = 0.7) -> Tuple[list[float], float]:
    """
    Implements the DE/rand/1/bin algorithm
    """
    # Create and initialize an dim-dimensional population
    population = [
        [random.uniform(min_x, max_x) for _ in range(dim)] for _ in range(n)
    ]
    fitness = [objective_func(i) for i in population]

    # Find initial best solution
    best_fitness = min(fitness)
    best_solution = list(population[fitness.index(best_fitness)])

    for generation in range(max_generations):
        for indiv in range(n):
            # Create the trial vector by applying the mutation operator
            indices = [idx for idx in range(n) if idx != indiv]
            i1, i2, i3 = random.sample(indices, 3)

            trial_vector = [
                population[i1][i]
                + scale_factor * (population[i2][i] - population[i3][i])
                for i in range(dim)
            ]

            # Clamp bounds
            trial_vector = [max(min_x, min(x, max_x)) for x in trial_vector]

            # Binomial crossover
            crossover_points = [
                random.random() < crossover_prob
                for _ in range(dim)
            ]
            if not any(crossover_points):
                j_rand = random.randint(0, dim - 1)
                crossover_points[j_rand] = True

            # Create an offspring by applying the crossover operator;
            offspring_vector = [
                trial_vector[i] if crossover_points[i] else population[indiv][i]
                for i in range(dim)
            ]

            # Evaluate fitness
            offspring_fitness = objective_func(offspring_vector)

            # add the offspring to the population if it is more fit
            if offspring_fitness < fitness[indiv]:
                population[indiv] = offspring_vector
                fitness[indiv] = offspring_fitness
                if offspring_fitness < best_fitness:
                    best_fitness = offspring_fitness
                    best_solution = offspring_vector

        # Optional: Print progress
        if generation % (max_generations // 10) == 0 \
            or generation == max_generations - 1:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.6f}")

    return best_solution, best_fitness

if __name__ == '__main__':
    dim = 2
    min_x = -5.12
    max_x = 5.12

    solution, fitness_value = de_rand_1_bin(dim, min_x, max_x, objective_func)

    print("\n--- Results ---")
    print(f"Best solution found: {solution}")
    print(f"Fitness of the best solution: {fitness_value}")
