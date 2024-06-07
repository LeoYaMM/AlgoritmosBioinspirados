'''
Resolver el problema de asignacion 
cuadratica usando ACO.
'''
import numpy as np
import itertools

def selecciona_siguiente_ciudad(actual, no_visitadas, feromona, distancia, alpha, beta, epsilon=1e-10):
    feromona_actual = feromona[actual][no_visitadas] ** alpha
    visibilidad = (1.0 / (distancia[actual][no_visitadas] + epsilon)) ** beta
    probabilidades = feromona_actual * visibilidad
    
    # Chequear y manejar valores infinitos o NaN en probabilidades
    if np.any(np.isinf(probabilidades)) or np.any(np.isnan(probabilidades)):
        probabilidades = np.nan_to_num(probabilidades, nan=epsilon, posinf=epsilon, neginf=epsilon)
    
    suma_probabilidades = sum(probabilidades)
    
    # Chequear si la suma de probabilidades es cero o infinito para evitar NaNs
    if suma_probabilidades == 0 or np.isinf(suma_probabilidades) or np.isnan(suma_probabilidades):
        probabilidades = np.ones_like(probabilidades) / len(probabilidades)
    else:
        probabilidades /= suma_probabilidades
    
    return np.random.choice(no_visitadas, 1, p=probabilidades)[0]

def construye_camino(inicio, n_ciudades, feromona, distancia, alpha, beta):
    camino = [inicio]
    no_visitadas = list(range(n_ciudades))
    no_visitadas.remove(inicio)

    actual = inicio
    while no_visitadas:
        siguiente = selecciona_siguiente_ciudad(actual, no_visitadas, feromona, distancia, alpha, beta)
        camino.append(siguiente)
        no_visitadas.remove(siguiente)
        actual = siguiente

    return camino

def computa_costo(camino, distancia, flujo):
    costo = 0
    for i in range(len(camino)):
        for j in range(len(camino)):
            costo += flujo[i][j] * distancia[camino[i]][camino[j]]
    return costo

def actualiza_feromona(feromona, caminos, costos, evaporacion, q):
    feromona *= (1.0 - evaporacion)
    for camino, costo in zip(caminos, costos):
        for i in range(len(camino) - 1):
            feromona[camino[i]][camino[i+1]] += q / costo
        feromona[camino[-1]][camino[0]] += q / costo

def aco(distancia, flujo, n_hormigas, n_iteraciones, evaporacion, alpha=1, beta=5, q=1):
    n_ciudades = distancia.shape[0]
    feromona = np.ones(distancia.shape) / n_ciudades

    mejor_camino = None
    mejor_costo = float('inf')

    for _ in range(n_iteraciones):
        caminos = [construye_camino(np.random.choice(n_ciudades), n_ciudades, feromona, distancia, alpha, beta) for _ in range(n_hormigas)]
        costos = [computa_costo(camino, distancia, flujo) for camino in caminos]
        actualiza_feromona(feromona, caminos, costos, evaporacion, q)
        min_costo = min(costos)
        if min_costo < mejor_costo:
            mejor_costo = min_costo
            mejor_camino = caminos[costos.index(min_costo)]

    return mejor_camino, mejor_costo

def leer_matrices(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    matrices = []
    current_matrix = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            current_matrix.append(list(map(int, stripped_line.split())))
        else:
            if current_matrix:
                matrices.append(np.array(current_matrix))
                current_matrix = []
    if current_matrix:
        matrices.append(np.array(current_matrix))
    
    return matrices

def ANOVA(distancia, flujo, n_iteraciones, evaporacion, alphas, betas, n_hormigas):
    resultados = []
    combinaciones = list(itertools.product(n_hormigas, alphas, betas))
    
    for n_h, alpha, beta in combinaciones:
        mejor_camino, mejor_costo = aco(distancia, flujo, n_h, n_iteraciones, evaporacion, alpha, beta)
        resultados.append((n_h, alpha, beta, mejor_camino, mejor_costo))
    
    return resultados

# Usar las matrices leídas
# ! Cambiar el path al archivo
# file_path = 'Algoritmos Bioinspirados/Practica 5/matricesProblemaQAP/matricesProblemaQAP/tai15.dat'
file_path = 'matricesProblemaQAP/matricesProblemaQAP/tai30.dat'
matrices = leer_matrices(file_path)
distancia = matrices[1]
flujo = matrices[2]

# Definir los parámetros
n_hormigas = [10, 25, 50, 100]
alphas = [50, 100, 200]
betas = [2, 10]
n_iteraciones = 100
evaporacion = 0.5

# Ejecutar experimentos
resultados = ANOVA(distancia, flujo, n_iteraciones, evaporacion, alphas, betas, n_hormigas)

# Imprimir resultados
for i, (n_h, alpha, beta, mejor_camino, costo) in enumerate(resultados):
    print(f"{i + 1}. Número de hormigas: {n_h}, alpha: {alpha}, beta: {beta}, Mejor camino: {mejor_camino}, Costo: {costo}")