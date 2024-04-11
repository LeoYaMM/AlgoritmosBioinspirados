import math
import random
import copy
import sys
import time
import matplotlib.pyplot as plt
import statistics

random.seed(2)

n = 3
nn = n * n
n2 = nn // 2
magic_number = n * (n ** 2 + 1) / 2
tamano_poblacion = 200
num_elites = 150
num_mutaciones = 0.9
probabilidad_mutacion = 1
epocas = 1000000
intentos_cruce = 20
tenencia_tabu = 10

poblacion = []
aptitud = []
lista_tabu = []

def generar_individuos():
    poblacion_generada = []
    for _ in range(tamano_poblacion):
        cuadrado = list(range(1, nn + 1))
        random.shuffle(cuadrado)
        poblacion_generada.append(cuadrado)
    return poblacion_generada

def calcular_aptitud_poblacional(poblacion_actual):
    aptitudes = []
    for individuo in poblacion_actual:
        aptitudes.append(calcular_aptitud_cuadrado(individuo))
    return aptitudes

def calcular_aptitud_cuadrado(cuadrado):
    aptitud = 0
    suma = 0
    for i in range(nn):
        if (i % n == 0 and i != 0):
            aptitud += abs(magic_number - suma)
            suma = 0
        suma += cuadrado[i]
    aptitud += abs(magic_number - suma)

    for j in range(n):
        suma = 0
        for i in range(j, nn, n):
            suma += cuadrado[i]
        aptitud += abs(magic_number - suma)

    suma = 0
    for i in range(0, nn, n + 1):
        suma += cuadrado[i]
    aptitud += abs(magic_number - suma)

    suma = 0
    for i in range(n - 1, nn - 1, n - 1):
        suma += cuadrado[i]
    aptitud += abs(magic_number - suma)

    return aptitud

def verificar_exito(aptitudes):
    if 0.0 in aptitudes:
        return aptitudes.index(0.0), 'exito'

    umbral_error_minimo = 5
    for indice, valor_aptitud in enumerate(aptitudes):
        if valor_aptitud <= umbral_error_minimo:
            return indice, 'error_minimo'

    return -1, 'no_encontrado'

def es_cuadrado_valido(cuadrado):
    numeros = set(range(1, nn + 1))
    cuadrado_posible = set(cuadrado)

    if numeros == cuadrado_posible:
        return True
    return False

def obtener_elites(poblacion_actual, aptitudes):
    combinados = list(zip(poblacion_actual, aptitudes))
    combinados = sorted(combinados, key=lambda x: x[1])
    elites, _ = zip(*combinados)
    mejor_aptitud = aptitudes[0]
    peor_aptitud = aptitudes[-1]
    favorito = sum(aptitudes) / len(aptitudes)
    desviacion_estandar = statistics.stdev(aptitudes)
    print(f"Mejor Aptitud: {mejor_aptitud}")
    print(f'Peor Aptitud: {peor_aptitud}')
    print(f'Aptitud Promedio: {favorito}')
    print(f'Desviacion Estandar: {desviacion_estandar}')
    print(f'Élite Top: {elites[0]}')
    return list(elites)

def mutar(cuadrado):
    numero_genes = len(cuadrado)
    numero_mutaciones = random.randint(1, numero_genes)

    for _ in range(numero_mutaciones):
        i = random.randint(0, numero_genes - 1)
        j = random.randint(0, numero_genes - 1)
        cuadrado[i], cuadrado[j] = cuadrado[j], cuadrado[i]

    return cuadrado

def reproducir(poblacion_actual, aptitudes):
    poblacion_copia = copy.deepcopy(poblacion_actual)
    pool_apareamiento = []
    siguiente_generacion = []

    individuos_restantes = tamano_poblacion - num_elites
    elites = obtener_elites(poblacion_actual, aptitudes)
    unicos_elites = []
    for elite in elites:
        if elite not in unicos_elites:
            unicos_elites.append(elite)

    siguiente_generacion = unicos_elites[0:num_elites]

    suma_aptitudes = 0
    inverso_aptitudes = []
    for aptitud in aptitudes:
        inverso_aptitudes.append(10000 - aptitud)
    for aptitud in inverso_aptitudes:
        suma_aptitudes += aptitud

    pool_apareamiento.append(inverso_aptitudes[0] / suma_aptitudes)
    for i in range(1, len(inverso_aptitudes)):
        probabilidad = inverso_aptitudes[i] / suma_aptitudes
        pool_apareamiento.append(probabilidad + pool_apareamiento[i - 1])

    while individuos_restantes > 0:
        eleccion1 = random.random()
        eleccion2 = random.random()
        indice1 = 0
        indice2 = 0
        for i in range(len(pool_apareamiento)):
            if eleccion1 <= pool_apareamiento[i]:
                indice1 = i
                break
        for i in range(len(pool_apareamiento)):
            if eleccion2 <= pool_apareamiento[i]:
                indice2 = i
                break

        candidato1 = copy.copy(poblacion_copia[indice1])
        candidato2 = copy.copy(poblacion_copia[indice2])

        hijo1, hijo2 = cruce(candidato1, candidato2)

        if (hijo1 not in siguiente_generacion) and (hijo1 not in lista_tabu):
            siguiente_generacion.append(hijo1)
            individuos_restantes -= 1
            if len(lista_tabu) >= tenencia_tabu:
                lista_tabu.pop(0)
            lista_tabu.append(hijo1)

        if ((hijo2 not in siguiente_generacion) and (hijo2 not in lista_tabu) and individuos_restantes > 0):
            siguiente_generacion.append(hijo2)
            individuos_restantes -= 1
            if len(lista_tabu) >= tenencia_tabu:
                lista_tabu.pop(0)
            lista_tabu.append(hijo2)

    print()
    return siguiente_generacion

def cruce(padre1, padre2):
    index1 = random.randint(0, nn - 1)
    index2 = random.randint(0, nn - 1)
    if index1 > index2:
        index1, index2 = index2, index1

    hijo1 = [None] * nn
    hijo2 = [None] * nn

    hijo1[index1:index2 + 1] = padre2[index1:index2 + 1]
    hijo2[index1:index2 + 1] = padre1[index1:index2 + 1]

    def completar_hijo(hijo, padre):
        pos_actual = (index2 + 1) % nn
        for gen in padre:
            if gen not in hijo:
                hijo[pos_actual] = gen
                pos_actual = (pos_actual + 1) % nn

    completar_hijo(hijo1, padre1)
    completar_hijo(hijo2, padre2)

    return hijo1, hijo2

poblacion = generar_individuos()
historial = {'mejor': [], 'peor': [], 'promedio': [], 'desviacion': []}

for i in range(epocas):
    print(f"Generación actual: {i}")
    aptitud = calcular_aptitud_poblacional(poblacion)

    historial['mejor'].append(min(aptitud))
    historial['peor'].append(max(aptitud))
    historial['promedio'].append(sum(aptitud) / len(aptitud))

    if i % 500 == 0 and i != 0:
        for j in range(len(poblacion)):
            print(f'Cuadrado: {poblacion[j]}  Aptitud: {aptitud[j]}')
        time.sleep(5)

    indice, estado = verificar_exito(aptitud)
    if estado == 'exito':
        print(f"Cuadrado Mágico Encontrado: {poblacion[indice]}")
        break
    elif estado == 'error_minimo':
        print(f"Cuadrado Mágico Encontrado: {poblacion[indice]} con Aptitud {aptitud[indice]}")
        break

    poblacion = reproducir(poblacion, aptitud)

generaciones = list(range(len(historial['mejor'])))

plt.scatter(generaciones, historial['mejor'], color='green', label='Mejor Aptitud')
plt.plot(generaciones, historial['mejor'], color='green')

plt.scatter(generaciones, historial['peor'], color='red', label='Peor Aptitud')
plt.plot(generaciones, historial['peor'], color='red')

plt.scatter(generaciones, historial['promedio'], color='blue', label='Aptitud Promedio')
plt.plot(generaciones, historial['promedio'], color='blue')

plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Aptitud')
plt.title("Gráfico de Convergencia")
plt.show()



random.seed(29)
n = 4
nn = n * n
n2 = nn // 2
magic_number = n * (n ** 2 + 1) / 2
population_size = 200
num_elites = 150
num_mutations = 0.9
mutation_chance = 1
epoch = 1000000
crossover_attempts = 20
tabu_tenure = 10

poblacion = []
aptitud = []
tabu_list = []

def generar_individuos():
    pop = []
    for i in range(population_size):
        c = list(range(1, nn + 1))
        random.shuffle(c)
        pop.append(c)
    return pop

def obtener_aptitud_poblacional(pop):
    fit = []
    for c in pop:
        fit.append(encontrar_aptitud2(c))
    return fit

def encontrar_aptitud2(criatura):
    fit = 0
    suma = 0
    for i in range(nn):
        if (i % n == 0 and i != 0):
            fit += abs(magic_number - suma)
            suma = 0
        suma += criatura[i]
    fit += abs(magic_number - suma)

    for j in range(n):
        suma = 0
        for i in range(j, nn, n):
            suma += criatura[i]
        fit += abs(magic_number - suma)

    suma = 0
    for i in range(0, nn, n + 1):
        suma += criatura[i]
    fit += abs(magic_number - suma)

    suma = 0
    for i in range(n - 1, nn - 1, n - 1):
        suma += criatura[i]
    fit += abs(magic_number - suma)

    return fit

def ganamos(apt):
    if 0.0 in apt:
        return apt.index(0.0), 'exito'

    umbral_error_minimo = 5
    for indice, valor_aptitud in enumerate(apt):
        if valor_aptitud <= umbral_error_minimo:
            return indice, 'error_minimo'

    return -1, 'no_encontrado'


def es_cuadrado_valido(criatura):
    numeros = set(range(1, nn + 1))
    cuadrado_posible = set(criatura)

    if numeros == cuadrado_posible:
        return True
    return False


def obtener_elites(pop, fit):
    combinados = list(zip(pop, fit))
    combinados = sorted(combinados, key=lambda x: x[1])
    elites, trash = zip(*combinados)
    mejor_aptitud = trash[0]
    peor_aptitud = trash[-1]
    favorito = sum(trash) / len(trash)
    desviacion = statistics.stdev(aptitud)
    print(f"Mejor Aptitud: {mejor_aptitud}")
    print(f'Peor Aptitud: {peor_aptitud}')
    print(f'Aptitud Promedio: {favorito}')
    print(f'Desviacion Estandar: {desviacion}')
    print(f'Élite Top: {elites[0]}')
    return list(elites)

def mutar(criatura):
    nn = len(criatura)
    num_mut = random.randint(1, nn)

    for _ in range(num_mut):
        i = random.randint(0, nn - 1)
        j = random.randint(0, nn - 1)
        criatura[i], criatura[j] = criatura[j], criatura[i]

    return criatura

def reproducir(pop, fit):
    popc = copy.deepcopy(pop)
    pool_apareamiento = []
    siguiente_gen = []

    numero_restante = population_size - num_elites
    elites = obtener_elites(pop, fit)
    unicos_elites = []
    for e in elites:
        if not e in unicos_elites:
            unicos_elites.append(e)

    siguiente_gen = unicos_elites[0:num_elites]

    fitsum = 0
    bigfit = []
    for i in aptitud:
        bigfit.append(10000 - i)

    for i in bigfit:
        fitsum += i

    pool_apareamiento.append(bigfit[0] / fitsum)
    for i in range(1, len(bigfit)):
        prob = bigfit[i] / fitsum
        pool_apareamiento.append(prob + pool_apareamiento[i - 1])

    while numero_restante > 0:
        eleccion1 = random.random()
        eleccion2 = random.random()
        indice1 = 0
        indice2 = 0
        for i in range(len(pool_apareamiento)):
            if eleccion1 <= pool_apareamiento[i]:
                indice1 = i
                break
        for i in range(len(pool_apareamiento)):
            if eleccion2 <= pool_apareamiento[i]:
                indice2 = i
                break

        candidato1 = copy.copy(popc[indice1])
        candidato2 = copy.copy(popc[indice2])

        bebe1, bebe2 = cruce(candidato1, candidato2)

        if (not bebe1 in siguiente_gen) and (not bebe1 in tabu_list):
            siguiente_gen.append(bebe1)
            numero_restante -= 1
            if len(tabu_list) >= tabu_tenure:
                tabu_list.pop(0)
            tabu_list.append(bebe1)

        if ((not bebe2 in siguiente_gen) and (not bebe2 in tabu_list) and numero_restante > 0):
            siguiente_gen.append(bebe2)
            numero_restante -= 1
            if len(tabu_list) >= tabu_tenure:
                tabu_list.pop(0)
            tabu_list.append(bebe2)

    print()
    return siguiente_gen

def cruce(p1, p2):
    nn = len(p1)
    index1 = random.randint(0, nn - 1)
    index2 = random.randint(0, nn - 1)
    if index1 > index2:
        index1, index2 = index2, index1

    c1 = [None] * nn
    c2 = [None] * nn

    c1[index1:index2 + 1] = p2[index1:index2 + 1]
    c2[index1:index2 + 1] = p1[index1:index2 + 1]

    def complete_child(child, parent):
        current_pos = (index2 + 1) % nn
        for gene in parent:
            if gene not in child:
                child[current_pos] = gene
                current_pos = (current_pos + 1) % nn

    complete_child(c1, p1)
    complete_child(c2, p2)

    return c1, c2

poblacion = generar_individuos()
historial_aptitud = {'mejor': [], 'peor': [], 'promedio': [], 'desviacion': []}

for i in range(epoch):
    print(f"Generación actual: {i}")
    aptitud = obtener_aptitud_poblacional(poblacion)

    historial_aptitud['mejor'].append(min(aptitud))
    historial_aptitud['peor'].append(max(aptitud))
    historial_aptitud['promedio'].append(sum(aptitud) / len(aptitud))


    if i % 500 == 0 and i != 0:
        for i in range(len(poblacion)):
            print(f'Cuadrado: {poblacion[i]}  Aptitud: {aptitud[i]}')
        time.sleep(5)

    indice, estado = ganamos(aptitud)
    if estado == 'exito':
        print(f"Cuadrado Mágico Encontrado: {poblacion[indice]}")
        break
    elif estado == 'error_minimo':
        print(f"Cuadrado Mágico Encontrado: {poblacion[indice]} con Aptitud {aptitud[indice]}")
        break

    poblacion = reproducir(poblacion, aptitud)

generaciones = list(range(len(historial_aptitud['mejor'])))

plt.scatter(generaciones, historial_aptitud['mejor'], color='green', label='Mejor Aptitud')
plt.plot(generaciones, historial_aptitud['mejor'], color='green')

plt.scatter(generaciones, historial_aptitud['peor'], color='red', label='Peor Aptitud')
plt.plot(generaciones, historial_aptitud['peor'], color='red')

plt.scatter(generaciones, historial_aptitud['promedio'], color='blue', label='Aptitud Promedio')
plt.plot(generaciones, historial_aptitud['promedio'], color='blue')

plt.legend()
plt.xlabel('Generaciones')
plt.ylabel('Aptitud')
plt.title("Gráfica de convergencia")
plt.show()

plt.show("prueba")