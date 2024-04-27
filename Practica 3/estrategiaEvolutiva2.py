# La estrategia evolutiva de este programa es (μ + λ) donde solo los descendientes son considerados para la siguiente generacion.
from matplotlib import pyplot as plt
import numpy as np
from functools import reduce
from operator import mul

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


# Estrategias evolutivas
def randSolution(interval, dimension):
    limInf, limSup = interval
    solution = np.round(np.random.uniform(limInf, limSup, dimension), 2)
    return solution

def clip(x, interval): # revisa los limites de las variables de decision
    limInf, limSup = interval
    return max(min(x, limSup), limInf)

def mutation(sigmaT, x, interval):
    newX = []
    for i in range(len(x)):
        newX.append(clip(np.round(x[i] + sigmaT * np.random.normal(0, 1), 2), interval))
    return newX

def crossover(parents, dimension):
    newX = []
    for i in range(dimension):
        index = np.random.randint(0, len(parents))
        newX.append(parents[index][i])
    return newX


def estrategiaEvolutiva(Gmax, dimension, interval, fun, mu, lamb, c, sigma):
    x = randSolution(interval, dimension) #solucion base/solución inicial (la podria tomar del conocimiento del problema)
    fx = fun(x)
    print('SOLUCION INICIAL', x, fx)
    bestSolution = []
    sigmas = []
    successes = 0
    ps = 0


    for gen in range(Gmax): # Numero maximo de generaciones
        if gen == 0:
            # Crea μ padres 
            parents = []
            for _ in range(mu):
                individual = mutation(sigma, x, interval)
                fitness = np.round(fun(individual), 4)
                parents.append([individual, fitness])
            print(f"Padres: {parents}")
        else:
            # Selecciona los nuevos padres a partir de los mejores individuos en la generación anterior
            # Ordena los individuos por su fitness y selecciona los mejores μ para ser padres
            parents = sorted(population, key=lambda child: child[1])[:mu]

        # Crear λ hijos a partir de los μ padres
        offspring = []
        for _ in range(lamb):
            # Seleccionar aleatoriamente a dos padres para el cruce
            # Generar índices aleatorios
            parent_indices = np.random.choice(len(parents), 2, replace=False)
            # Usar índices para obtener los individuos
            p1, p2 = parents[parent_indices[0]][0], parents[parent_indices[1]][0]
            child = crossover([p1, p2], dimension)
            child = mutation(sigma, child, interval)
            offspring.append([child, np.round(fun(child), 4)])

        population = parents + offspring
        for _ in range(len(population)):
            # Actualizar la mejor solución
            if population[-1][1] < fun(x):
                x = population[-1][0]
                successes += 1

        print(f"Generacion {gen} Descendientes:\n{offspring}")

        bestSolution.append(fun(x)) 
        sigmas.append(sigma)

        #Actualizar ps: frecuencia relativa de mutaciones exitosas.
        if gen % (10 * dimension) == 0: # calcula la proporción de éxito cada 10*n generaciones
            ps = successes / (gen + 1)
            #if gen%n == 0: # n mas grande  ps se mantiene mas generaciones 
            if ps > 1/5:
                sigma = sigma / c # no hay tantos éxitos por lo tanto explora regiones con tamaños de paso más grande 
            elif ps < 1/5:
                sigma = sigma * c  # encuentra región prometedora por lo tanto refina la solución actual(explotacion)
            elif ps == 1/5: # caso contrario sigma queda con el mismo valor
                sigma = sigma
        
    return bestSolution, sigmas

Gmax = 1000
np.random.seed(38)
dimension = 10
fun = rosenbrock

interval = (-2.048, 2.048) if fun == rosenbrock else \
            (-32.768, 32.768) if fun == ackley else \
            (-600, 600) if fun == griewank else \
            (-5.12, 5.12) if fun == rastrigin else None

mu = 20
lamb = 30 
sigma = 0.5 if fun == rosenbrock else \
            2.0 if fun == ackley else \
            20 if fun == griewank else \
            1.0 if fun == rastrigin else None
c = 0.817 

best, sigmas = estrategiaEvolutiva(Gmax, dimension, interval, fun, mu, lamb, c, sigma)

print('BEST', best)
print('SIGMAS', sigmas)
plt.plot(range(0, len(best)), best, color = 'green', label='mejores')
plt.legend()
plt.show()
plt.plot(range(0, len(sigmas)), sigmas, label='sigmas')
plt.legend()

plt.show()
