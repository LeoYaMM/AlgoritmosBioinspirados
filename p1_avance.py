import numpy as np
import random

# Define the Rosenbrock function
def rosenbrock(x):
    resultado = 0
    for i in range(len(x) - 1):
        resultado += 100 * ((x[i] ** 2) - x[i + 1]) ** 2 + (1 - x[i]) ** 2
    return resultado


# Define the Ackley function
def ackley(x):
    return - 20 * np.exp(-0.2 * np.sqrt(sum(xi**2 for xi in x) / len(x))) - np.exp(sum([np.cos(2 * np.pi * xi) for xi in x]) / len(x)) + 20 + np.exp(1)


puntos_rosenbrock = [round(random.uniform(-2.048, 2.048), 2) for _ in range(1, 11)]
puntos_evaluados_rosenbrock = round(rosenbrock(puntos_rosenbrock), 4)
puntos_ackley = [round(random.uniform(-32.768, 32.768), 2) for _ in range(1, 11)]
puntos_evaluados_ackley = round(ackley(puntos_ackley), 4)

print(f"Puntos seleccionados para rosenbrock: {puntos_rosenbrock}")
print(f"Resultado rosenbrock: {puntos_evaluados_rosenbrock}")
print(f"Puntos seleccionados para ackley: {puntos_ackley}")
print(f"Resultado ackley: {puntos_evaluados_ackley}")
