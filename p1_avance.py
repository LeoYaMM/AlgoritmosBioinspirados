import numpy as np
import random

def summation(x):
    sum = 0
    for xi in x:
        sum += xi

    return sum


# Define the Rosenbrock function
def rosenbrock(x):
    resultado = 0
    for i, punto in enumerate(x):
        resultado  += 100 * ((punto**2) - x[i + 1])**2 + (1 - punto)**2
        if i < len(x):
            break

    return resultado

# Define the Ackley function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x**2) / len(x)))
    cos_term = -np.exp(sum(np.cos(c * x) / len(x)))
    return sum_sq_term + cos_term + a + np.exp(1)


poblacion = [a for a in range(1, 11)]
puntos_rosenbrock = [round(random.uniform(-2.048, 2.048), 2) for _ in range(1, 11)]
puntos_evaluados = round(rosenbrock(puntos_rosenbrock), 4)

print(f"Puntos seleccionados: {puntos_rosenbrock}")
print(f"Resultado: {puntos_evaluados}")