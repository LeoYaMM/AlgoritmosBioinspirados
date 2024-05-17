# Regresion lineal simple con PSO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
file_path = 'Salary.csv'
data = pd.read_csv(file_path)

# Preparar los datos
X = data['YearsExperience'].values
y = data['Salary'].values

# Definir la función de error cuadrático (MSE corregido)
def mse(params):
    a, b = params
    y_pred = a + b * X
    return np.sum((y - y_pred) ** 2)


def pso(num_particles, dimension, objective_function, w, c1, c2, generations, L, U):
    # Asignamos a n_a la función np.array para no tener que escribir np.array continuamente
    nA = np.array
    # Inicializar la posición y la velocidad de las partículas
    particles = [nA([np.random.uniform(L, U) for _ in range(dimension)]) for _ in range(num_particles)]
    velocities = [nA([0 for _ in range(dimension)]) for _ in range(num_particles)]

    # Establecemos el primer pbest y gbest
    pbest = particles
    gbest = min(pbest, key=objective_function)
    print(f"Posicion inicial de las partículas: {particles}")
    print(f"Mejor posición global inicial: {gbest} fitness: {objective_function(gbest)}")

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
    
    print(f"Particulas: {particles}")
    print(f"Mejor posición por partícula: {pbest}")
    print(f"Mejor posición global: {gbest}")
    print(f"Fitness de la mejor posicion: {objective_function(gbest)}") 
    # print(f"Fitness promedio de las mejores posiciones por particula: {np.mean([objective_function(p) for p in pbest])}")

    return gbest


# Parámetros
num_particles = 100
dimension = 2
# ω={0.25,0.5}
w = 0.5
# c1={0.0,1.0,2.0}
c1 = 2
# c2={0.0,1.0,2.0}
c2 = 2
generations = 500
fun = mse

# Definir límites uniformes para los parámetros a y b
L = 30000
U = 50000

# Se establece la semilla para reproducibilidad
np.random.seed(0)

gbest = pso(num_particles, dimension, fun, w, c1, c2, generations, L, U)

a, b = gbest
print(f"Mejor solución encontrada: a = {a}, b = {b}")

# Calcular valores predichos
y_pred = a + b * X

# Graficar los datos y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred, color='red', label='Regresión lineal')
plt.title('Regresión lineal simple con PSO')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.legend()
plt.grid(True)
plt.show()