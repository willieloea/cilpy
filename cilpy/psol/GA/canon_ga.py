from typing import Tuple, List
import random

def objective_func(position: list[float]) -> float:
    """
    The Sphere function (objective function).
    f(x) = sum(x_i^2) for i=1 to n
    The global minimum is at x = (0, ..., 0) with f(x) = 0.
    """
    return sum(x**2 for x in position)

def tournament_selection(population: List[list[float]],
                         fitness: List[float],
                         k: int) -> list[float]:
    """
    Selects a parent from the population using tournament selection.
    """
    # Select k random individuals
    tournament_indices = random.sample(range(len(population)), k)
    
    # Find the best individual
    best_index = tournament_indices[0]
    for i in tournament_indices[1:]:
        if fitness[i] < fitness[best_index]:
            best_index = i
            
    return population[best_index]


def simulated_binary_crossover(parent1: list[float],
                               parent2: list[float],
                               eta: float,
                               prob: float) -> Tuple[list[float], list[float]]:
    """
    Performs simulated binary crossover (SBX) on two parents.

    Args:
        parent1: The first parent.
        parent2: The second parent.
        eta: The crowding degree of the crossover. A higher eta creates
             offspring closer to the parents.
        prob: The probability of performing crossover for each variable.

    Returns:
        A tuple containing the two offspring.
    """
    offspring1 = list(parent1)
    offspring2 = list(parent2)
    dim = len(parent1)

    for i in range(dim):
        if random.random() > prob:
            continue

        # Calculate beta
        r = random.random()
        if r <= 0.5:
            gamma = (2 * r)**(1.0 / (eta + 1.0))
        else:
            gamma = (1.0 / (2.0 * (1.0 - r)))**(1.0 / (eta + 1.0))

        # Apply crossover
        offspring1[i] = 0.5 * ((1+gamma) * parent1[i] + (1-gamma) * parent2[i])
        offspring2[i] = 0.5 * ((1-gamma) * parent1[i] + (1+gamma) * parent2[i])

    return offspring1, offspring2


def uniform_mutation(individual: list[float],
                     prob: float,
                     min_x: float,
                     max_x: float) -> list[float]:
    """
    Performs uniform mutation on an individual.

    Args:
        individual: The individual to mutate.
        prob: The probability of mutating each variable.
        min_x: The lower bound for the variable's value.
        max_x: The upper bound for the variable's value.

    Returns:
        The mutated individual.
    """
    mutated_indiv = list(individual)
    dim = len(individual)

    for i in range(dim):
        if random.random() < prob:
            if random.randint(0, 1) == 0:
                # x_ij(t) = x_ij(t) + delta(t, x_max,j - x_ij(t))
                delta = random.uniform(0, max_x - mutated_indiv[i])
                mutated_indiv[i] += delta
            else:
                # x_ij(t) = x_ij(t) - delta(t, x_ij(t) - x_min,j)
                delta = random.uniform(0, mutated_indiv[i] - min_x)
                mutated_indiv[i] -= delta
            
            mutated_indiv[i] = max(min_x, min(mutated_indiv[i], max_x))

    return mutated_indiv


def genetic_algorithm(dim: int,
                      min_x: float,
                      max_x: float,
                      objective_func: callable,
                      selection_mec: callable,
                      crossover_mec: callable,
                      mutation_mec: callable,
                      n: int = 100,
                      max_generations: int = 500,
                      crossover_prob: float = 0.9,
                      mutation_prob: float = 0.1,
                      crossover_eta: float = 20.0,
                      tournament_size: int = 3) -> Tuple[list[float], float]:
    """
    Implements a Genetic Algorithm.
    """
    # Initialize Population
    population = [
        [random.uniform(min_x, max_x) for _ in range(dim)] for _ in range(n)
    ]
    fitness = [objective_func(i) for i in population]
    
    # Find initial best solution
    best_fitness = min(fitness)
    best_solution = list(population[fitness.index(best_fitness)])

    for generation in range(max_generations):
        # Create new offspring
        offspring_population = []
        
        while len(offspring_population) < n:
            # Selection
            parent1 = selection_mec(population, fitness, tournament_size)
            parent2 = selection_mec(population, fitness, tournament_size)

            # Crossover
            offspring1, offspring2 = crossover_mec(parent1, parent2,
                                                   crossover_eta,
                                                   crossover_prob)
            
            # Mutation
            offspring1 = mutation_mec(offspring1, mutation_prob, min_x, max_x)
            offspring2 = mutation_mec(offspring2, mutation_prob, min_x, max_x)
            
            offspring_population.extend([offspring1, offspring2])

        # Combine offspring and parents
        offspring_fitness = [objective_func(ind) for ind in \
                             offspring_population]
        combined_population = population + offspring_population
        combined_fitness = fitness + offspring_fitness
        sorted_indices = sorted(range(len(combined_fitness)),
                                key=lambda k: combined_fitness[k])
        
        # Select the best n individuals for the next generation
        population = [combined_population[i] for i in sorted_indices[:n]]
        fitness = [combined_fitness[i] for i in sorted_indices[:n]]

        # Update the global best
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_solution = list(population[0])
            
        # Optional: Print progress
        if generation % (max_generations // 10) == 0 \
            or generation == max_generations - 1:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.6f}")

    return best_solution, best_fitness

if __name__ == '__main__':
    dim = 10
    min_x = -5.12
    max_x = 5.12

    solution, fitness_value = genetic_algorithm(
        dim=dim,
        min_x=min_x,
        max_x=max_x,
        objective_func=objective_func,
        selection_mec=tournament_selection,
        crossover_mec=simulated_binary_crossover,
        mutation_mec=uniform_mutation
    )

    print("\n--- Results ---")
    print(f"Best solution found: {solution}")
    print(f"Fitness of the best solution: {fitness_value}")
