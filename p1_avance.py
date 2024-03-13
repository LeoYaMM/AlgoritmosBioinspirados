import numpy as np
import random

# Define the Rosenbrock function
def rosenbrock(x):
    return sum(100 * (x- x**2.0)**2.0 + (1 - x)**2.0)

# Define the Ackley function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x**2) / len(x)))
    cos_term = -np.exp(sum(np.cos(c * x) / len(x)))
    return sum_sq_term + cos_term + a + np.exp(1)


poblacion = [a for a in range(1, 11)]

# Bounds for Rosenbrock function, as given in the problem statement
rosenbrock_bounds = [(-2.048, 2.048)] * 10

# Bounds for Ackley function, as given in the problem statement
ackley_bounds = [(-32.768, 32.768)] * 10

# Optimize Rosenbrock function
rosen_result = minimize(rosenbrock, initial_guess, bounds=rosenbrock_bounds, method='L-BFGS-B')

# Optimize Ackley function
ackley_result = minimize(ackley, initial_guess, bounds=ackley_bounds, method='L-BFGS-B')

rosen_result.x, ackley_result.x