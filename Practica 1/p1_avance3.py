# Codificación binaria para números reales
import numpy
import p1_avance
import random

def seleccionDePadre(poblacion, fitness):
    total_fitness = sum(fitness)
    # Generar un número aleatorio entre 0 y la suma total de fitness
    seleccion = random.uniform(0, total_fitness)
    # Recorrer la población y acumular el fitness hasta que se alcance el valor aleatorio
    acumulado = 0
    for i, individuo in enumerate(poblacion):
        acumulado += fitness[i]
        if acumulado >= seleccion:
            print(f"Valor acumulado: {acumulado}")
            return individuo


def cruza(padre1, padre2):
    #funcion de cruza mediante 2 puntos
    
    return

def mutacion():
    #funcion
    return

def rosenbrock(x):
    resultado = 0
    for i in range(len(x) - 1):
        resultado += 100 * ((x[i] ** 2) - x[i + 1]) ** 2 + (1 - x[i]) ** 2
    return resultado

def seleccionDeSobrevivientes(individuos, competencias):
    valores_objetivo = {ind: rosenbrock for ind, x in individuos.items()}
    sobrevivientes = []
    for comp in competencias:
        ganador = comp[0] if valores_objetivo[comp[0]] > valores_objetivo[comp[1]] else comp[1]
        sobrevivientes.append(ganador)
    return sobrevivientes
    
def generar_individuos():
    individuos = {}
    for i in range(1, 11):
        x = (random.uniform(-40, 40),2)
        y = (random.uniform(-40, 40),2)
        individuos[i] = (x, y)       
    return individuos

def generar_competencias(n_competencias, n_individuos):
    competencias = []
    for _ in range(n_competencias):
        competidores = random.sample(range(1, n_individuos + 1), 2)  
        competencias.append(tuple(competidores))
    return competencias

individuos_aleatorios = generar_individuos()
competencias = generar_competencias(5, 10)
sobrevivientes = seleccionDeSobrevivientes(individuos_aleatorios, competencias)

print("\nResultados de las competiciones (sobrevivientes):")
for i, competencia in enumerate(competencias):
    print(f"[{competencia[0]}] vs [{competencia[1]}] -> Sobreviviente: [{sobrevivientes[i]}]")