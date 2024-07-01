import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

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
num_vertices = 10 # Numero de nodos
max_iter = 100

# Probabilidad de conexión
p = 0.3  # Ajusta esta probabilidad para obtener la densidad deseada

# Crear el grafo
graph = nx.erdos_renyi_graph(num_vertices, p)

# Asegurarse de que el grafo no sea totalmente conexo
while nx.is_connected(graph):
    graph = nx.erdos_renyi_graph(num_vertices, p)

# Dibujar el grafo
plt.figure(figsize=(8, 6))
nx.draw(graph, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
plt.show()

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

# Dibujar el grafo final coloreado
color_map = [f"C{color}" for color in gbest_position]  # Asigna un color a cada nodo según la solución óptima
plt.figure(figsize=(8, 6))
nx.draw(graph, with_labels=True, node_color=color_map, node_size=500, edge_color='gray')
plt.show()

