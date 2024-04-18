import numpy as np
import matplotlib.pyplot as plt

# Configuración básica
n = 3
nn = n * n
magicNumber = n * (n**2 + 1) // 2
populationSize = 100
elites = 20
mutationRate = 0.05
epochs = 1000

# Función para generar la población inicial
def generate_population(size):
    population = []
    for _ in range(size):
        individual = np.random.permutation(nn) + 1
        population.append(individual)
    return np.array(population)

# Función para calcular la aptitud de un individuo
def fitness(individual):
    matrix = individual.reshape((n, n))
    fitness = 0
    fitness += np.abs(np.sum(matrix, axis=0) - magicNumber).sum()
    fitness += np.abs(np.sum(matrix, axis=1) - magicNumber).sum()
    fitness += abs(np.sum(np.diag(matrix)) - magicNumber)
    fitness += abs(np.sum(np.diag(np.fliplr(matrix))) - magicNumber)
    return fitness

# Función para seleccionar los mejores individuos
def select_elites(population, fitnesses, num_elites):
    # Argsort devuelve los índices que ordenarían el array, y seleccionamos los `num_elites` primeros
    elite_indices = np.argsort(fitnesses)[:num_elites]
    return population[elite_indices]


# Función para realizar el cruce entre dos individuos
def crossover(parent1, parent2):
    cross_point = np.random.randint(nn)
    child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
    unique, counts = np.unique(child, return_counts=True)
    duplicates = unique[counts > 1]
    missing = np.setdiff1d(np.arange(1, nn+1), unique)
    np.random.shuffle(missing)
    for duplicate in duplicates:
        dupe_indices = np.where(child == duplicate)[0]
        child[dupe_indices[1:]] = missing[:len(dupe_indices)-1]
        missing = missing[len(dupe_indices)-1:]
    return child

# Función para mutar un individuo
def mutate(individual):
    if np.random.rand() < mutationRate:
        swap_indices = np.random.choice(nn, 2, replace=False)
        individual[swap_indices[0]], individual[swap_indices[1]] = individual[swap_indices[1]], individual[swap_indices[0]]
    return individual

# Ejecutar algoritmo evolutivo
num_elites = 20  # Más descriptivo y evita confusión más adelante
population = generate_population(populationSize)
history = {'best': []}

for epoch in range(epochs):
    fitnesses = np.array([fitness(ind) for ind in population])
    history['best'].append(fitnesses.min())
    if fitnesses.min() == 0:
        print("Magic square found!")
        print(population[np.argmin(fitnesses)].reshape((n, n)))
        break
    elite_individuals = select_elites(population, fitnesses, num_elites)  # Usando la nueva variable num_elites
    new_population = [crossover(elite_individuals[i % num_elites], elite_individuals[(i+1) % num_elites]) for i in range(populationSize - num_elites)]
    new_population = [mutate(ind) for ind in new_population]
    population = np.vstack((elite_individuals, new_population))


# Plot de resultados
plt.plot(history['best'], label='Best Fitness')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Fitness')
plt.title('Evolution of Fitness')
plt.show()
