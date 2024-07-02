import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# Funcion de fitness
def fitness(solution, graph):
    conflicts = 0
    for edge in graph.edges:
        if solution[edge[0]] == solution[edge[1]]:
            conflicts += 1
    num_colors = len(set(solution))
    return num_colors + conflicts * 1000

# PSO parameters
num_particles = 30
num_nodos = 10 # Numero de nodos
max_iter = 100

# Probabilidad de conexion
p = 0.3

# Crear el grafo
graph = nx.erdos_renyi_graph(num_nodos, p)

# Asegurarse de que el grafo no sea totalmente conexo
while nx.is_connected(graph):
    graph = nx.erdos_renyi_graph(num_nodos, p)

# Dibujar el grafo
plt.figure(figsize=(8, 6))
nx.draw(graph, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
plt.show()

# Inicializacion de particulas
particles = [np.random.randint(0, num_nodos, num_nodos) for _ in range(num_particles)]
velocities = [np.random.randint(-1, 2, num_nodos) for _ in range(num_particles)]
pbest_positions = particles.copy()
pbest_scores = [fitness(p, graph) for p in particles]
gbest_position = pbest_positions[np.argmin(pbest_scores)]
gbest_score = min(pbest_scores)

# Parametros de PSO
w = 0.5  # Inercia
c1 = 1.0  # Constante cognitiva
c2 = 1.0  # Constante social

# PSO main loop
for iteration in range(max_iter):
    for i in range(num_particles):
        velocities[i] = (w * velocities[i]
                         + c1 * random.random() * (pbest_positions[i] - particles[i])
                         + c2 * random.random() * (gbest_position - particles[i]))
        
        # Actualiza la posicion de las particulas
        particles[i] = np.clip(particles[i] + velocities[i], 0, num_nodos - 1).astype(int)
        
        # Evaluar nueva posicion
        current_fitness = fitness(particles[i], graph)
        
        # Actualizar pbest
        if current_fitness < pbest_scores[i]:
            pbest_positions[i] = particles[i]
            pbest_scores[i] = current_fitness
        
        # Actualizar gbest
        if current_fitness < gbest_score:
            gbest_position = particles[i]
            gbest_score = current_fitness
    
    print(f"Iteracion {iteration + 1}/{max_iter}, Mejor Fitness: {gbest_score}")

# Resultado final
print("Mejor solucion encontrada:", gbest_position)
print("Numero de colores utilizados:", len(set(gbest_position)))

# Dibujar el grafo final coloreado
color_map = [f"C{color}" for color in gbest_position]  # Asigna un color a cada nodo segun la solucipn optima
plt.figure(figsize=(8, 6))
nx.draw(graph, with_labels=True, node_color=color_map, node_size=500, edge_color='gray')
plt.show()

