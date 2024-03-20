# Minimizar las funciones
# Codificacion real
import numpy as np
import random
from operator import itemgetter

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

    return children

def genetic(f, dimension, lower_limit, upper_limit, poblation_size, crossover_percentage, mutation_percentage, generations):
    # Inicializar la población
    poblation = start_poblation(f, poblation_size, dimension, lower_limit, upper_limit)
    print(f"Poblacion inicial: {poblation}")
    # Listas para la grafica de convergencia
    best, worse, average = [], [], []

    for generation in range(generations):
        print(f"Generacion {generation + 1}")
        # Seleccionar parejas
        couples = define_couples(poblation, poblation_size)
        print(f"Parejas seleccionadas: {couples}")

        # Generar hijos
        children = create_children(couples, poblation, crossover_percentage)
        print(f"Hijos generados: {children}")

        # Generar mutaciones
        mutated_children = []
        for child in children:
            mutated = mutation(child, 0.1, mutation_percentage)
            if f == "rosenbrock":
                mutated_children.append([mutated, rosenbrock(mutated)])
            elif f == "ackley":
                mutated_children.append([mutated, ackley(mutated)])

        print(f"Hijos mutados: {mutated_children}")

        # Unir la población con los hijos
        new_poblation = poblation + mutated_children
        print(f"Nueva poblacion: {new_poblation}")

        # Ordenar la población
        sorted_poblation = sorted(new_poblation, key=itemgetter(1))
        print(f"Poblacion ordenada: {sorted_poblation}")

        # Seleccionar los mejores individuos
        poblation = sorted_poblation[:poblation_size]
        print(f"Mejores individuos: {poblation}")

        # Calcular el mejor, peor y promedio de la población
        best.append(poblation[0][1])
        worse.append(poblation[-1][1])
        average.append(sum([ind[1] for ind in poblation]) / poblation_size)

    print(f"Mejor individuo: {poblation[0]}")
    
    return best, worse, average

def plot_convergence(best, worse, average, generations):
    import matplotlib.pyplot as plt

    X = list(range(1, generations + 1))
    plt.scatter(X, best, color="green", label="Mejor")
    plt.scatter(X, worse, color="red", label="Peor")
    plt.scatter(X, average, color="blue", label="Promedio")
    plt.plot(X, best, color="green")
    plt.plot(X, worse, color="red")
    plt.plot(X, average, color="blue")
    plt.xlabel("Generaciones")
    plt.ylabel("Aptitud")
    plt.title("Grafica de convergencia")
    plt.legend()
    plt.show()



op = True

while op:
    print("Funcion a resolver: ")
    print("1. Rosenbrock")
    print("2. Ackley")
    op = int(input("Opcion: "))
    if op == 1:
        lower_limit = -2.048
        upper_limit = 2.048
        f = "rosenbrock"
        break
    elif op == 2:
        lower_limit = -32.768
        upper_limit = 32.768
        f = "ackley"
        break
    else:
        print("Debe elegir una opcion valida")

poblation_size = int(input("Tamaño de la poblacion: "))
dimension = int(input("Dimension: "))
generations = int(input("Generaciones: "))
crossover_percentage = float(input("Porcentaje de cruza: "))
mutation_percentage = float(input("Porcentaje de mutacion: "))

# Semilla para obtener los mismos resultados
random.seed(12)

best, worse, average = genetic(f, dimension, lower_limit, upper_limit, poblation_size, crossover_percentage, mutation_percentage, generations)
plot_convergence(best, worse, average, generations)