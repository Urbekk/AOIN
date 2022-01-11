import random
import numpy as np


def check_fitness(population, values, weights, n_items, max_weight):
    fitness = []

    for j, chromosome in enumerate(population):
        while True:

            cur_weight = 0
            cur_value = 0

            for i in range(0, n_items):
                if chromosome & pow(2, i) != 0:
                    cur_weight += weights[n_items - i - 1]
                    cur_value += values[n_items - i - 1]

            if cur_weight <= max_weight:
                fitness.append(cur_value)
                break

            else:
                remove = ~pow(2, random.randint(1, n_items))
                chromosome &= remove

        population[j] = chromosome

    return fitness


def generate_population(population_size, n_items):
    population = []

    for i in range(0, population_size):
        population.append(random.randint(0, pow(2, n_items) - 1))

    return population


def select_parents(population, fitness):
    target_fitness = random.randint(0, sum(fitness))
    temp_sum = 0
    parent1 = []
    fitness1 = 0
    selected = ([], [])

    for i in range(0, len(population)):
        temp_sum += fitness[i]

        if temp_sum >= target_fitness:
            parent1 = population.pop(i)
            fitness1 = fitness.pop(i)
            break

    temp_sum = 0
    target_fitness = random.randint(0, sum(fitness))

    for i in range(0, len(population)):
        temp_sum += fitness[i]

        if temp_sum >= target_fitness:
            parent2 = population[i]
            population.append(parent1)
            fitness.append(fitness1)
            selected = (parent1, parent2)
            break

    return selected


def crossover(parent1, parent2, n_items):
    pivot = random.randint(0, n_items - 1)
    max_value = pow(2, n_items) - 1
    mask1 = (max_value >> pivot) << pivot
    mask2 = max_value - mask1

    child = (parent1 & mask1) + (parent2 & mask2)

    return child


def mutate(chromosome, n_items, mutation_resillence):
    for i in range(0, n_items - 1):
        mutation_cofficient = random.randint(1, mutation_resillence)
        if mutation_cofficient == 1:
            chromosome ^= pow(2, i)

    return chromosome


def get_elite(population, fitness):
    elite = 0

    for i, cur_fitness in enumerate(fitness):
        if cur_fitness > fitness[elite]:
            elite = i

    return population[elite]


def generate_next_generation(population, fitness, n_items, mutation_resillence):
    population_size = len(fitness)
    new_population = [get_elite(population, fitness)]

    while (len(new_population) < population_size):
        (parent1, parent2) = select_parents(population, fitness)
        new_population.append(mutate(crossover(parent1, parent2, n_items), n_items, mutation_resillence))

    return new_population


def is_converged(fitness, convergence_acceptance):
    mode_fitness = max(set(fitness), key=fitness.count)

    if float(fitness.count(mode_fitness)) / float(len(fitness)) >= convergence_acceptance:
        return True

    return False


def binary_to_array(number, n_items):
    array = [int(x) for x in format(number, "0{}b".format(n_items))]

    return array


def solve_knapsack(values, weights, max_weight, population_size, mutation_resillence, generations, acceptance):
    generation = 0
    population = generate_population(population_size, len(values))
    fitness = check_fitness(population, values, weights, len(values), max_weight)

    while (not is_converged(fitness, acceptance) and generation < generations):
        generation += 1
        population = generate_next_generation(population, fitness, len(values), mutation_resillence)
        fitness = check_fitness(population, values, weights, len(values), max_weight)

    return (generation, get_elite(population, fitness))


def run(max_weight, values, weights):
    population_size = 50

    (generations, solution) = solve_knapsack(values, weights, max_weight, population_size, 100, 100, 0.6)
    solution_arr = binary_to_array(solution, len(values))

    cur_val = 0
    cur_weight = 0

    for i, x in enumerate(solution_arr):

        if x == 1:
            cur_val += values[i]
            cur_weight += weights[i]

    # print("Solution found in {} generations".format(generations))
    # print("Selection is: ", solution_arr)
    # print("Total values is: ", cur_val)
    # print("Total weight is: ", cur_weight)

    return cur_weight, np.count_nonzero(solution_arr)

