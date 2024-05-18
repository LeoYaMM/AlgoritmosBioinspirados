# Particle swarm optimization

import numpy as np
from operator import mul
from functools import reduce
import matplotlib.pyplot as plt

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


def pso(num_particles, dimension, objective_function, w, c1, c2, generations, L, U):
    # Asignamos a n_a la función np.array para no tener que escribir np.array continuamente
    nA = np.array
    # Inicializar la posición y la velocidad de las partículas
    particles = [nA([np.random.uniform(L, U) for _ in range(dimension)]) for _ in range(num_particles)]
    velocities = [nA([0 for _ in range(dimension)]) for _ in range(num_particles)]
    
    # Listas para almacenar el mejor y peor fitness en cada generación
    best_fitness_per_gen = []
    worst_fitness_per_gen = []

    # Establecemos el primer pbest y gbest
    pbest = particles
    gbest = min(pbest, key=objective_function)
    print(f"Posicion inicial de las partículas: {particles}")
    print(f"Mejor posición global inicial: {gbest} fitness: {objective_function(gbest)}")

    best_fitness_per_gen.append(objective_function(gbest))
    worst_fitness_per_gen.append(max([objective_function(p) for p in particles]))

    for _ in range(generations):
        for i in range(num_particles):
            r1, r2 = np.random.uniform(0, 1, 2)
            # Actualizamos la velocidad
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            # Actualizamos la posición
            particles[i] = particles[i] + velocities[i]
            # Actualizamos el pbest
            if objective_function(particles[i]) < objective_function(pbest[i]):
                pbest[i] = particles[i]

        # Actualizamos el gbest
        gbest = min(pbest, key=objective_function)

        # Registrar el mejor y peor fitness de la generación actual
        fitness_values = [objective_function(p) for p in particles]
        best_fitness_per_gen.append(objective_function(gbest))
        worst_fitness_per_gen.append(max(fitness_values))
    
    print(f"Particulas: {particles}")
    print(f"Mejor posición por partícula: {pbest}")
    print(f"Mejor posición global: {gbest}")
    print(f"Fitness de la mejor posicion: {objective_function(gbest)}") 
    print(f"Fitness promedio de las mejores posiciones por particula: {np.mean([objective_function(p) for p in pbest])}")

    # Generar la gráfica de convergencia
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_per_gen, marker='o', linestyle='-', color='b', label='Mejor ejecución')
    plt.plot(worst_fitness_per_gen, marker='s', linestyle='-', color='g', label='Peor ejecución')
    plt.title(f'Gráfica de convergencia {objective_function.__name__}')
    plt.xlabel('Generaciones')
    plt.ylabel('Función Objetivo f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Parámetros
num_particles = 100
dimension = 10
#ω={0.25,0.5}
w = 0.5
# c1={0.0,1.0,2.0}
c1 = 2.0
# c2={0.0,1.0,2.0}
c2 = 2.0
generations = 5000
fun = rosenbrock

L, U = (-2.048, 2.048) if fun == rosenbrock else \
        (-32.768, 32.768) if fun == ackley else \
        (-600, 600) if fun == griewank else \
        (-5.12, 5.12) if fun == rastrigin else None

# Se establece la semilla para reproducibilidad
np.random.seed(0)

pso(num_particles, dimension, fun, w, c1, c2, generations, L, U)