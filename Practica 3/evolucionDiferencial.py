import numpy as np
from functools import reduce
from operator import mul
import os
# import estrategiaEvolutiva1 as ee1

def clear_screen():
    # Comprueba si el sistema operativo es Windows
    if os.name == 'nt':
        os.system('cls')  # cls es el comando para limpiar la consola en Windows
    else:
        os.system('clear')  # clear es el comando para limpiar la consola en Unix/Linux

# Funciones a optimizar
def rosenbrock(x):
    result = 0
    for i in range(len(x) - 1):
        result += 100 * ((x[i] ** 2) - x[i + 1]) ** 2 + (1 - x[i]) ** 2
    return result

def ackley(x):
    summation1 = sum(xi**2 for xi in x)
    summation2 = sum(np.cos(2 * np.pi * xi) for xi in x)
    exp1 = np.exp(-0.2 * np.sqrt((1 / len(x)) * summation1))
    exp2 = np.exp((1 / len(x)) * summation2)
    result = (- 20) * exp1 - exp2 + np.e + 20
    return result

def griewank(x):
    summation = sum(xi**2 for xi in x)
    producer = reduce(mul, np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    result = (1 / 4000) * summation - producer + 1
    return result

def rastrigin(x):
    summation = sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)
    result = summation + 10 * len(x)
    return result


# Definir el algoritmo de Evolución Diferencial
def differential_evolution(strategy, objective_function, dimensions, population_size, F, R, Gmax, L, U):
    # Ejemplo para 'rand/1/bin':
    if strategy == 'rand/1/bin':
        X, best = rand_1_bin(objective_function, dimensions, population_size, F, R, Gmax, L, U)
        print(X)
        print(f"Best: {best}")
        
    # Ejemplo para 'rand/1/exp':
    elif strategy == 'rand/1/exp':
        X, best = rand_1_exp(objective_function, dimensions, population_size, F, R, Gmax, L, U)
        print(X)
        print(f"Best: {best}")
        
    # Ejemplo para 'best/1/bin':
    elif strategy == 'best/1/bin':
        X, best = best_1_bin(objective_function, dimensions, population_size, F, R, Gmax, L, U)
        print(X)
        print(f"Best: {best}")
        
    # Ejemplo para 'best/1/exp':
    elif strategy == 'best/1/exp':
        X, best = best_1_exp(objective_function, dimensions, population_size, F, R, Gmax, L, U)
        print(X)
        print(f"Best: {best}")
        
    return

#crea poblacion dentro de los intervalos de la funcion objetivo
def start_poblation(population_size, dimension, lower_limit, upper_limit):
    poblation = []

    for _ in range(population_size):
        individual = []
        for _ in range(dimension):
            individual.append(np.random.uniform(lower_limit, upper_limit))
        
        poblation.append(individual)

    return poblation

# Generacion de r1 != r2 != r3
def random_numbers(population_size, n):
    indexes = np.random.randint(0, population_size, n)
    while len(set(indexes)) != n:
        indexes = np.random.randint(0, population_size, n)
    return indexes

def clip(x, interval): # revisa los limites de las variables de decision
    limInf, limSup = interval
    return max(min(x, limSup), limInf)

# Ejecutar el experimento
def run_experiment(strategy, objective_function, dimensions, population_size, F, R, Gmax, runs, L, U):
    for _ in range(runs):
        differential_evolution(strategy, objective_function, dimensions, population_size, F, R, Gmax, L, U)
    return

def rand_1_bin(objective_function, dimensions, population_size, F, R, Gmax, L, U):
    X = start_poblation(population_size, dimensions, L, U)
    print("============================Rand/1/bin=============================")
    print(f"===================Funcion objetivo: {objective_function.__name__}====================")
    print("=========================Poblacion inicial=========================")
    print(f"{X} tamaño: {len(X)}")
    best_individual = min(X, key=objective_function)
    best = objective_function(best_individual)
    for g in range(Gmax):
        for i in range(population_size):
            r1, r2, r3 = random_numbers(population_size, 3)
            jrand = np.random.randint(0, dimensions)
            ui = []
            for j in range(dimensions):
                if np.random.rand() < R or j == jrand:
                    u = X[r1][j] + F * (X[r2][j] - X[r3][j])
                    u = clip(u, (L, U))
                    ui.append(u)
                else:
                    ui.append(X[i][j])
            
            if objective_function(ui) <= objective_function(X[i]):
                X[i] = ui
            
            if objective_function(X[i]) < best:
                best = objective_function(X[i])
                best_individual = X[i]
        print(f"Generacion {g} Mejor individuo: {best_individual} Fitness: {best}")

    return X, best

def rand_1_exp(objective_function, dimensions, population_size, F, R, Gmax, L, U):
    X = start_poblation(population_size, dimensions, L, U)
    print("============================Rand/1/exp=============================")
    print(f"===================Funcion objetivo: {objective_function.__name__}====================")
    print("=========================Poblacion inicial=========================")
    print(f"{X} tamaño: {len(X)}")
    best_individual = min(X, key=objective_function)
    best = objective_function(best_individual)
    for g in range(Gmax):
        for i in range(population_size):
            r1, r2, r3 = random_numbers(population_size, 3)
            jrand = np.random.randint(0, dimensions)
            ui = X[i][:]
            j = 0
            # n = 0
            while True:
                u = X[r1][j] + F * (X[r2][j] - X[r3][j])
                u = clip(u, (L, U))
                ui[j] = u
                j += 1 #% dimensions  # Incrementa j de manera circular
                # jrand = (jrand + 1) % dimensions  # Asegura que al menos uno cambie
                # n += 1
                if j < dimensions and np.random.rand() < R or j == jrand:
                    break

            if objective_function(ui) <= objective_function(X[i]):
                X[i] = ui
            
            if objective_function(X[i]) < best:
                best = objective_function(X[i])
                best_individual = X[i]
        print(f"Generacion {g} Mejor individuo: {best_individual} Fitness: {best}")

    return X, best

def best_1_bin(objective_function, dimensions, population_size, F, R, Gmax, L, U):
    X = start_poblation(population_size, dimensions, L, U)
    print("============================Best/1/bin=============================")
    print(f"===================Funcion objetivo: {objective_function.__name__}====================")
    print("=========================Poblacion inicial=========================")
    print(f"{X} tamaño: {len(X)}")
    best_individual = min(X, key=objective_function)
    best = objective_function(best_individual)
    for g in range(Gmax):
        for i in range(population_size):
            r2, r3 = random_numbers(population_size, 2)
            jrand = np.random.randint(0, dimensions)
            ui = []
            for j in range(dimensions):
                if np.random.rand() < R or j == jrand:
                    u = best_individual[j] + F * (X[r2][j] - X[r3][j])
                    u = clip(u, (L, U))
                    ui.append(u)
                else:
                    ui.append(X[i][j])
            
            if objective_function(ui) <= objective_function(X[i]):
                X[i] = ui
            
            if objective_function(X[i]) < best:
                best = objective_function(X[i])
                best_individual = X[i]
        print(f"Generacion {g} Mejor individuo: {best_individual} Fitness: {best}")

    return X, best

def best_1_exp(objective_function, dimensions, population_size, F, R, Gmax, L, U):
    X = start_poblation(population_size, dimensions, L, U)
    print("============================Best/1/exp=============================")
    print(f"===================Funcion objetivo: {objective_function.__name__}====================")
    print("=========================Poblacion inicial=========================")
    print(f"{X} tamaño: {len(X)}")
    best_individual = min(X, key=objective_function)
    best = objective_function(best_individual)
    for g in range(Gmax):
        for i in range(population_size):
            r2, r3 = random_numbers(population_size, 2)
            jrand = np.random.randint(0, dimensions)
            ui = X[i][:]
            j = 0
            # n = 0
            while True:
                u = best_individual[j] + F * (X[r2][j] - X[r3][j])
                u = clip(u, (L, U))
                ui[j] = u
                j = (j + 1) #% dimensions  # Incrementa j de manera circular
                # jrand = (jrand + 1) % dimensions  # Asegura que al menos uno cambie
                # n += 1
                if np.random.rand() < R or j == jrand:
                    break

            if objective_function(ui) <= objective_function(X[i]):
                X[i] = ui
            
            if objective_function(X[i]) < best:
                best = objective_function(X[i])
                best_individual = X[i]
        print(f"Generacion {g} Mejor individuo: {best_individual} Fitness: {best}")

    return X, best

# Configuración de parámetros según la imagen
population_size = 50
dimensions = 10
F = 0.6
R = 0.9
Gmax = 1000
runs = 1

np.random.seed(7)

# Estrategias de evolución diferencial
strategies = ['rand/1/bin', 'rand/1/exp', 'best/1/bin', 'best/1/exp']

# Funciones objetivo
objective_functions = [rosenbrock, ackley, griewank, rastrigin]

# Correr los experimentos
for strategy in strategies:
    for objective_function in objective_functions:
        if objective_function == rosenbrock:
            interval = (-2.048, 2.048)
        elif objective_function == ackley:
            interval = (-32.768, 32.768)
        elif objective_function == griewank:
            interval = (-600, 600)
        elif objective_function == rastrigin:
            interval = (-5.12, 5.12)
        lower_limit, upper_limit = interval
        run_experiment(strategy, objective_function, dimensions, population_size, F, R, Gmax, runs, lower_limit, upper_limit)
        input("Presione Enter para continuar...")
        clear_screen()
