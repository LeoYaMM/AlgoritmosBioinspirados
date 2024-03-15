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

def seleccionDeSobrevivientes():
    #funcion
    return

