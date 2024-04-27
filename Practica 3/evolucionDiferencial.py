import numpy as np
import matplotlib.pyplot as plt
import estrategiaEvolutiva1 as ee1

# Definir el algoritmo de Evolución Diferencial
def differential_evolution(strategy, objective_function, dimensions, population_size, F, Cr, Gmax, interval):
    # Inicializar población y evaluarla
    population = ee1.randSolution(interval, dimensions)
    for generation in range(Gmax):
        new_population = []
        for i, target in enumerate(population):
            # Seleccionar agentes para las operaciones de mutación y recombinación basado en la estrategia
            # Ejemplo para 'rand/1/bin':
            if strategy == 'rand/1/bin':
                ...
            # Ejemplo para 'rand/1/exp':
            elif strategy == 'rand/1/exp':
                ...
            # Ejemplo para 'best/1/bin':
            elif strategy == 'best/1/bin':
                ...
            # Ejemplo para 'best/1/exp':
            elif strategy == 'best/1/exp':
                ...
            # Realizar la recombinación y la selección
            ...
        population = new_population
    return min(population, key=objective_function)



# Ejecutar el experimento
def run_experiment(strategy, objective_function, dimensions, population_size, F, Cr, Gmax, runs):
    best_solutions = []
    for _ in range(runs):
        best_solution = differential_evolution(strategy, objective_function, dimensions, population_size, F, Cr, Gmax)
        best_solutions.append(objective_function(best_solution))
    return np.mean(best_solutions), np.std(best_solutions)

# Configuración de parámetros según la imagen
population_size = 50
dimensions = 10
F = 0.6
Cr = 0.9
Gmax = 1000
runs = 20

np.random.seed(123)

# Estrategias de evolución diferencial
strategies = ['rand/1/bin', 'rand/1/exp', 'best/1/bin', 'best/1/exp']

# Funciones objetivo
objective_functions = [ee1.rosenbrock, ee1.ackley, ee1.griewank, ee1.rastrigin]

# Correr los experimentos
for strategy in strategies:
    for objective_function in objective_functions:
        if objective_function == ee1.rosenbrock:
            interval = (-2.048, 2.048)
        elif objective_function == ee1.ackley:
            interval = (-32.768, 32.768)
        elif objective_function == ee1.griewank:
            interval = (-600, 600)
        elif objective_function == ee1.rastrigin:
            interval = (-5.12, 5.12)
        mean, std_dev = run_experiment(strategy, objective_function, dimensions, population_size, F, Cr, Gmax, runs, interval)
        print(f'Strategy: {strategy}, Function: {objective_function.__name__}, Mean: {mean}, Std Dev: {std_dev}')
