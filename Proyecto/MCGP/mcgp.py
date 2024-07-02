import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import itertools

# Funcion de fitness
def fitness(solution, graph):
    conflicts = 0
    for edge in graph.edges:
        if solution[edge[0]] == solution[edge[1]]:
            conflicts += 1
    num_colors = len(set(solution))
    return num_colors + conflicts * 1000

def ANOVA(num_particles, w, c1, c2, num_combinations=5):
    combinaciones = list(itertools.product(num_particles, w, c1, c2))
    random.shuffle(combinaciones)  # Mezcla las combinaciones
    return combinaciones[:num_combinations]  # Devuelve las primeras `num_combinations` combinaciones

# PSO parameters
num_particles = [30, 50, 100]
num_nodos = 10 # Numero de nodos
max_iter = 100

# Parametros de PSO
w = [0.3, 0.5, 0.8]  # Inercia
c1 = [1.0, 1.5, 2.0]  # Constante cognitiva
c2 = [1.0, 1.5, 2.0]  # Constante social

# Probabilidad de conexion
p = 0.5

# Crear el grafo
graph = nx.erdos_renyi_graph(num_nodos, p)

# Asegurarse de que el grafo no sea totalmente conexo
while nx.is_connected(graph):
    graph = nx.erdos_renyi_graph(num_nodos, p)

# Dibujar el grafo
plt.figure(figsize=(8, 6))
nx.draw(graph, with_labels=True, node_color='gray', node_size=500, edge_color='black')
plt.show()

# Generar las combinaciones de parametros
parametros = ANOVA(num_particles, w, c1, c2, num_combinations=5)

for i, (num_particles, w, c1, c2) in enumerate(parametros):
    print(f"Experimento {i + 1}: num_particles={num_particles}, w={w}, c1={c1}, c2={c2}")

    # Inicializacion de particulas
    particles = [np.random.randint(0, num_nodos, num_nodos) for _ in range(num_particles)]
    velocities = [np.random.randint(-1, 2, num_nodos) for _ in range(num_particles)]
    pbest_positions = particles.copy()
    pbest_scores = [fitness(p, graph) for p in particles]
    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = min(pbest_scores)

    # PSO main loop
    for iteration in range(max_iter):
        for j in range(num_particles):
            velocities[j] = (w * velocities[j]
                            + c1 * random.random() * (pbest_positions[j] - particles[j])
                            + c2 * random.random() * (gbest_position - particles[j]))
            
            # Actualiza la posicion de las particulas
            particles[j] = np.clip(particles[j] + velocities[j], 0, num_nodos - 1).astype(int)
            
            # Evaluar nueva posicion
            current_fitness = fitness(particles[j], graph)
            
            # Actualizar pbest
            if current_fitness < pbest_scores[j]:
                pbest_positions[j] = particles[j]
                pbest_scores[j] = current_fitness
            
            # Actualizar gbest
            if current_fitness < gbest_score:
                gbest_position = particles[j]
                gbest_score = current_fitness
        
        print(f"Iteracion {iteration + 1}/{max_iter}, Mejor Fitness: {gbest_score}")

    # Resultado final
    print("Mejor solucion encontrada:", gbest_position)
    print("Numero de colores utilizados:", len(set(gbest_position)))
    print("\n\n")


    # Dibujar el grafo final coloreado para cada experimento
    color_map = [f"C{color}" for color in gbest_position]  # Asigna un color a cada nodo segun la solucipn optima
    plt.figure(figsize=(8, 6))
    nx.draw(graph, with_labels=True, node_color=color_map, node_size=500, edge_color='gray')
    plt.show()
