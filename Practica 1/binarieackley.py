#Minimizar las funciones

import numpy as np
import random
from operator import itemgetter

def float_to_binary(value, precision=10):
    # Convertir a entero aproximado para preservar la precisión
    int_value = int(value * (10 ** precision))
    # Convertir a binario y retornar
    return bin(int_value)[2:]

# Función auxiliar para convertir listas de flotantes a binario
def list_to_binary(list, precision=10):
    return [float_to_binary(val, precision) for val in list]

# Define the Rosenbrock function
def rosenbrock(x):
    resultado = 0
    for i in range(len(x) - 1):
        resultado += 100 * ((x[i] ** 2) - x[i + 1]) ** 2 + (1 - x[i]) ** 2
    return resultado

# Define the Ackley function
def ackley(x):
    return - 20 * np.exp(-0.2 * np.sqrt(sum(xi**2 for xi in x) / len(x))) - np.exp(sum([np.cos(2 * np.pi * xi) for xi in x]) / len(x)) + 20 + np.exp(1)

def start_poblation(f, poblation_size, dimension, lower_limit, upper_limit):
    poblation = []

    for _ in range(poblation_size):
        individual = []
        for _ in range(dimension):
            individual.append(random.uniform(lower_limit, upper_limit))
        
        if f == "rosenbrock":
            evaluation = rosenbrock(individual)
        elif f == "ackley":
            evaluation = ackley(individual)
        
        poblation.append([individual, evaluation])
        
    print(f"Poblacion inicial: {[{'individuo': list_to_binary(ind[0]), 'evaluacion': float_to_binary(ind[1])} for ind in poblation]}")
    
    return poblation

def define_couples(poblacion, poblacion_size):
    fitness = [poblacion[i][1] for i in range(poblacion_size)]
    selected = []

    # Calcular la suma total de fitness
    total_fitness = sum(fitness)

    while len(selected) < poblacion_size:
        # Generar un número aleatorio entre 0 y la suma total de fitness
        selection = random.uniform(0, total_fitness)
        
        acumulated = 0
        for i, fit in enumerate(fitness):
            acumulated += fit
            # Verificar si el acumulado supera el número aleatorio y si el índice no ha sido seleccionado previamente
            if acumulated >= selection:
                if i not in selected:
                    selected.append(i)
                    break
                else:
                    # Si el índice ya fue seleccionado, generar un nuevo número aleatorio y reiniciar el proceso
                    selection = random.uniform(0, total_fitness)
                    acumulated = 0

    # Asegurar que la cantidad de índices seleccionados sea par para formar parejas
    if len(selected) % 2 != 0:
        selected.pop()
        
    print(f"Parejas seleccionadas: {[bin(sel) for sel in selected]}")

    return selected

def crossover(parent1, parent2):
    # Seleccionar un punto de cruce aleatorio evitando extremos
    crossover_point = random.randint(1, len(parent1) - 1)

    # Sumamos las listas padre para obtener las listas hijas
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2

def mutation(child, delta, mutation_percentage):
    mutated_child = []

    for gen in child:
        if random.uniform(0, 1) <= mutation_percentage:
            if random.uniform(0, 1) <= 0.5:
                gen += delta
            else:
                gen -= delta
        
        mutated_child.append(gen)

    return mutated_child

def create_children(couples, poblation, crossover_percentage):
    children = []
    i = 0
    
    while i < len(couples):
        if random.uniform(0, 1) <= crossover_percentage:
            parent1 = poblation[couples[i]][0]
            parent2 = poblation[couples[i + 1]][0]
            child1, child2 = crossover(parent1, parent2)
            children.append(child1)
            children.append(child2)
        
        i += 2

    print(f"Hijos generados: [{', '.join([str({'individuo': list_to_binary(child)}) for child in children])}]")
    return children

def genetic(f, dimension, lower_limit, upper_limit, poblation_size, crossover_percentage, mutation_percentage, generations):
    # Inicializar la población
    poblation = start_poblation(f, poblation_size, dimension, lower_limit, upper_limit)
    # No es necesario imprimir de nuevo la población inicial aquí ya que se imprime dentro de start_poblation

    # Listas para la gráfica de convergencia
    best, worse, average = [], [], []

    for generation in range(generations):
        print(f"Generacion {generation + 1}")

        # Seleccionar parejas
        couples = define_couples(poblation, poblation_size)

        # Generar hijos
        children = create_children(couples, poblation, crossover_percentage)

        # Generar mutaciones
        mutated_children = []
        for child in children:
            mutated = mutation(child, 0.1, mutation_percentage)
            if f == "rosenbrock":
                mutated_children.append([mutated, rosenbrock(mutated)])
            elif f == "ackley":
                mutated_children.append([mutated, ackley(mutated)])

        # Convertir la evaluación de los hijos mutados a binario para la impresión
        print(f"Hijos mutados: [{', '.join([str({'individuo': list_to_binary(mut[0]), 'evaluacion': float_to_binary(mut[1])}) for mut in mutated_children])}]")

        # Unir la población con los hijos
        new_poblation = poblation + mutated_children

        # Ordenar la población
        sorted_poblation = sorted(new_poblation, key=itemgetter(1))

        # Seleccionar los mejores individuos
        poblation = sorted_poblation[:poblation_size]

        # Calcular el mejor, peor y promedio de la población para cada generación
        best.append(poblation[0][1])
        worse.append(poblation[-1][1])
        average.append(sum([ind[1] for ind in poblation]) / poblation_size)

    # Convertir la información del mejor individuo de la última generación a binario para la impresión
    print(f"Mejor individuo final: {{'individuo': {list_to_binary(poblation[0][0])}, 'evaluacion': {float_to_binary(poblation[0][1])}}}")
    
    return best, worse, average


random.seed(12)

f = "ackley"  # Opción 1 seleccionada
poblation_size = 3
dimension = 2
generations = 3
crossover_percentage = 0.9
mutation_percentage = 0.7

# Límites específicos para la función Rosenbrock
lower_limit = -32.768
upper_limit = 32.768

best, worse, average = genetic(f, dimension, lower_limit, upper_limit, poblation_size, crossover_percentage, mutation_percentage, generations)