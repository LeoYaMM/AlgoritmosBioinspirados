# Minimum Coloring Graph Problem
# Solución para el problema: PSO
import numpy as np
import networkx as nx
import random

# Función de fitness
def fitness(solution, graph):
    conflicts = 0
    for edge in graph.edges:
        if solution[edge[0]] == solution[edge[1]]:
            conflicts += 1
    num_colors = len(set(solution))
    return num_colors + conflicts * 1000  # Penalización por conflictos

# PSO parameters
num_particles = 30
num_vertices = 10  # Cambia esto según el tamaño de tu grafo
max_iter = 100
graph = nx.erdos_renyi_graph(num_vertices, 0.5)  # Ejemplo de un grafo aleatorio

# Inicialización de partículas
particles = [np.random.randint(0, num_vertices, num_vertices) for _ in range(num_particles)]
velocities = [np.random.randint(-1, 2, num_vertices) for _ in range(num_particles)]
pbest_positions = particles.copy()
pbest_scores = [fitness(p, graph) for p in particles]
gbest_position = pbest_positions[np.argmin(pbest_scores)]
gbest_score = min(pbest_scores)

# Parámetros de PSO
w = 0.5  # Inercia
c1 = 1.0  # Constante cognitiva
c2 = 1.0  # Constante social

# PSO main loop
for iteration in range(max_iter):
    for i in range(num_particles):
        velocities[i] = (w * velocities[i]
                         + c1 * random.random() * (pbest_positions[i] - particles[i])
                         + c2 * random.random() * (gbest_position - particles[i]))
        
        # Actualiza la posición de las partículas
        particles[i] = np.clip(particles[i] + velocities[i], 0, num_vertices - 1).astype(int)
        
        # Evaluar nueva posición
        current_fitness = fitness(particles[i], graph)
        
        # Actualizar pbest
        if current_fitness < pbest_scores[i]:
            pbest_positions[i] = particles[i]
            pbest_scores[i] = current_fitness
        
        # Actualizar gbest
        if current_fitness < gbest_score:
            gbest_position = particles[i]
            gbest_score = current_fitness
    
    print(f"Iteración {iteration + 1}/{max_iter}, Mejor Fitness: {gbest_score}")

# Resultado final
print("Mejor solución encontrada:", gbest_position)
print("Número de colores utilizados:", len(set(gbest_position)))
