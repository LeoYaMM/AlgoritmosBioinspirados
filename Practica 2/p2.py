import math
import random
import copy
import sys
import time
import matplotlib.pyplot as plt
import statistics

random.seed(23)

n = 3
nn = n * n
n2 = nn // 2
magicNumber = n * (n ** 2 + 1) / 2
poblationSize = 200
elites = 150
alfaMutaciones = 0.9
probMutacion = 1
epocas = 1000000
intentos_cruce = 20
tenencia_tabu = 10

poblacion = []
aptitud = []
tabu = []

def generarIndividuos():
    populationGen = []
    for _ in range(poblationSize):
        cuadrado = list(range(1, nn + 1))
        random.shuffle(cuadrado)
        populationGen.append(cuadrado)
    return populationGen

def aptitudPoblacional(poblacionActual):
    aptitudes = []
    for individuo in poblacionActual:
        aptitudes.append(calcularAptitudCuadrado(individuo))
    return aptitudes

def calcularAptitudCuadrado(cuadrado):
    aptitud = 0
    suma = 0
    for i in range(nn):
        if (i % n == 0 and i != 0):
            aptitud += abs(magicNumber - suma)
            suma = 0
        suma += cuadrado[i]
    aptitud += abs(magicNumber - suma)

    for j in range(n):
        suma = 0
        for i in range(j, nn, n):
            suma += cuadrado[i]
        aptitud += abs(magicNumber - suma)

    suma = 0
    for i in range(0, nn, n + 1):
        suma += cuadrado[i]
    aptitud += abs(magicNumber - suma)

    suma = 0
    for i in range(n - 1, nn - 1, n - 1):
        suma += cuadrado[i]
    aptitud += abs(magicNumber - suma)

    return aptitud

def verificar_exito(aptitudes):
    if 0.0 in aptitudes:
        return aptitudes.index(0.0), 'exito'

    error = 5
    for indice, aptitud in enumerate(aptitudes):
        if aptitud <= error:
            return indice, 'error_minimo'

    return -1, 'no_encontrado'

def es_cuadrado_valido(cuadrado):
    numeros = set(range(1, nn + 1))
    cuadradoPosible = set(cuadrado)

    if numeros == cuadradoPosible:
        return True
    return False

def fronteras(poblacionActual, aptitudes):
    combinados = list(zip(poblacionActual, aptitudes))
    combinados = sorted(combinados, key=lambda x: x[1])
    elites, _ = zip(*combinados)
    mejorAptitud = aptitudes[0]
    peorAptitud = aptitudes[-1]
    mejor = sum(aptitudes) / len(aptitudes)
    dessviacionSTD = statistics.stdev(aptitudes)
    print(f"Mejor Aptitud: {mejorAptitud}")
    print(f'Peor Aptitud: {peorAptitud}')
    print(f'Aptitud Promedio: {mejor}')
    print(f'Desviacion Estandar: {dessviacionSTD}')
    print(f'Élite Top: {elites[0]}')
    return list(elites)

def mutar(cuadrado):
    numGenes = len(cuadrado)
    numMutaciones = random.randint(1, numGenes)

    for _ in range(numMutaciones):
        i = random.randint(0, numGenes - 1)
        j = random.randint(0, numGenes - 1)
        cuadrado[i], cuadrado[j] = cuadrado[j], cuadrado[i]

    return cuadrado

def reproducir(poblacionActual, aptitudes):
    poblacion2 = copy.deepcopy(poblacionActual)
    apareamiento = []
    generacion2 = []

    individuosRestantes = poblationSize - elites
    elites = fronteras(poblacionActual, aptitudes)
    unicos_elites = []
    for elite in elites:
        if elite not in unicos_elites:
            unicos_elites.append(elite)

    generacion2 = unicos_elites[0:elites]

    sumaAptitudes = 0
    inversoAptitudes = []
    for aptitud in aptitudes:
        inversoAptitudes.append(10000 - aptitud)
    for aptitud in inversoAptitudes:
        sumaAptitudes += aptitud

    apareamiento.append(inversoAptitudes[0] / sumaAptitudes)
    for i in range(1, len(inversoAptitudes)):
        probabilidad = inversoAptitudes[i] / sumaAptitudes
        apareamiento.append(probabilidad + apareamiento[i - 1])

    while individuosRestantes > 0:
        eleccion1 = random.random()
        eleccion2 = random.random()
        indice1 = 0
        indice2 = 0
        for i in range(len(apareamiento)):
            if eleccion1 <= apareamiento[i]:
                indice1 = i
                break
        for i in range(len(apareamiento)):
            if eleccion2 <= apareamiento[i]:
                indice2 = i
                break

        candidato1 = copy.copy(poblacion2[indice1])
        candidato2 = copy.copy(poblacion2[indice2])

        hijo1, hijo2 = crossover(candidato1, candidato2)

        if (hijo1 not in generacion2) and (hijo1 not in tabu):
            generacion2.append(hijo1)
            individuosRestantes -= 1
            if len(tabu) >= tenencia_tabu:
                tabu.pop(0)
            tabu.append(hijo1)

        if ((hijo2 not in generacion2) and (hijo2 not in tabu) and individuosRestantes > 0):
            generacion2.append(hijo2)
            individuosRestantes -= 1
            if len(tabu) >= tenencia_tabu:
                tabu.pop(0)
            tabu.append(hijo2)

    print()
    return generacion2

def crossover(padre1, padre2):
    index1 = random.randint(0, nn - 1)
    index2 = random.randint(0, nn - 1)
    if index1 > index2:
        index1, index2 = index2, index1

    hijo1 = [None] * nn
    hijo2 = [None] * nn

    hijo1[index1:index2 + 1] = padre2[index1:index2 + 1]
    hijo2[index1:index2 + 1] = padre1[index1:index2 + 1]

    def completarHijo(hijo, padre):
        posicion = (index2 + 1) % nn
        for gen in padre:
            if gen not in hijo:
                hijo[posicion] = gen
                posicion = (posicion + 1) % nn

    completarHijo(hijo1, padre1)
    completarHijo(hijo2, padre2)

    return hijo1, hijo2

poblacion = generarIndividuos()
historial = {'mejor': [], 'peor': [], 'promedio': [], 'desviacion': []}

for i in range(epocas):
    print(f"Generación actual: {i}")
    aptitud = aptitudPoblacional(poblacion)

    historial['mejor'].append(min(aptitud))
    historial['peor'].append(max(aptitud))
    historial['promedio'].append(sum(aptitud) / len(aptitud))

    if i % 500 == 0 and i != 0:
        for j in range(len(poblacion)):
            print(f'Cuadrado: {poblacion[j]}  Aptitud: {aptitud[j]}')
        time.sleep(3)

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



random.seed(23)
n = 4
nn = n * n
n2 = nn // 2
magicNumber = n * (n ** 2 + 1) / 2
tamanioPoblacion = 200
elites = 150
num_mututacion = 0.9
mutationChance = 1
epoch = 1000000
intentoCruza = 20
tabu2 = 10

poblacion = []
aptitud = []
listaTabu = []

def generarIndividuos():
    pop = []
    for _ in range(tamanioPoblacion):
        c = list(range(1, nn + 1))
        random.shuffle(c)
        pop.append(c)
    return pop

def obtenerAptitudPoblacional(pop):
    fit = []
    for c in pop:
        fit.append(encontrarAptitud2(c))
    return fit

def encontrarAptitud2(hijo):
    fit = 0
    suma = 0
    for i in range(nn):
        if (i % n == 0 and i != 0):
            fit += abs(magicNumber - suma)
            suma = 0
        suma += hijo[i]
    fit += abs(magicNumber - suma)

    for j in range(n):
        suma = 0
        for i in range(j, nn, n):
            suma += hijo[i]
        fit += abs(magicNumber - suma)

    suma = 0
    for i in range(0, nn, n + 1):
        suma += hijo[i]
    fit += abs(magicNumber - suma)

    suma = 0
    for i in range(n - 1, nn - 1, n - 1):
        suma += hijo[i]
    fit += abs(magicNumber - suma)

    return fit

def wins(apt):
    if 0.0 in apt:
        return apt.index(0.0), 'exito'

    umbral_error_minimo = 5
    for indice, valor_aptitud in enumerate(apt):
        if valor_aptitud <= umbral_error_minimo:
            return indice, 'error_minimo'

    return -1, 'no_encontrado'


def esCuadradoValido(criatura):
    numeros = set(range(1, nn + 1))
    cuadrado_posible = set(criatura)

    if numeros == cuadrado_posible:
        return True
    return False


def obtenerElites(pop, fit):
    combinados = list(zip(pop, fit))
    combinados = sorted(combinados, key=lambda x: x[1])
    elites, trash = zip(*combinados)
    betterApt = trash[0]
    worstApt = trash[-1]
    elBueno = sum(trash) / len(trash)
    desviacion = statistics.stdev(aptitud)
    print(f"Mejor Aptitud: {betterApt}")
    print(f'Peor Aptitud: {worstApt}')
    print(f'Aptitud Promedio: {elBueno}')
    print(f'Desviacion Estandar: {desviacion}')
    print(f'Élite Top: {elites[0]}')
    return list(elites)

def mutar(hijo):
    nn = len(hijo)
    num_mut = random.randint(1, nn)

    for _ in range(num_mut):
        i = random.randint(0, nn - 1)
        j = random.randint(0, nn - 1)
        hijo[i], hijo[j] = hijo[j], hijo[i]

    return hijo

def reproducir(pop, fit):
    popc = copy.deepcopy(pop)
    apareamiento = []
    siguienteGen = []

    numeroRestante = tamanioPoblacion - elites
    elites = fronteras(pop, fit)
    elites = []
    for e in elites:
        if not e in elites:
            elites.append(e)

    siguienteGen = elites[0:elites]

    fitsum = 0
    bigfit = []
    for i in aptitud:
        bigfit.append(10000 - i)

    for i in bigfit:
        fitsum += i

    apareamiento.append(bigfit[0] / fitsum)
    for i in range(1, len(bigfit)):
        prob = bigfit[i] / fitsum
        apareamiento.append(prob + apareamiento[i - 1])

    while numeroRestante > 0:
        eleccion1 = random.random()
        eleccion2 = random.random()
        indice1 = 0
        indice2 = 0
        for i in range(len(apareamiento)):
            if eleccion1 <= apareamiento[i]:
                indice1 = i
                break
        for i in range(len(apareamiento)):
            if eleccion2 <= apareamiento[i]:
                indice2 = i
                break

        candidato1 = copy.copy(popc[indice1])
        candidato2 = copy.copy(popc[indice2])

        hijo1, hijo2 = crossover(candidato1, candidato2)

        if (not hijo1 in siguienteGen) and (not hijo1 in listaTabu):
            siguienteGen.append(hijo1)
            numeroRestante -= 1
            if len(listaTabu) >= tabu2:
                listaTabu.pop(0)
            listaTabu.append(hijo1)

        if ((not hijo2 in siguienteGen) and (not hijo2 in listaTabu) and numeroRestante > 0):
            siguienteGen.append(hijo2)
            numeroRestante -= 1
            if len(listaTabu) >= tabu2:
                listaTabu.pop(0)
            listaTabu.append(hijo2)

    print()
    return siguienteGen

def crossover(padre1, padre2):
    nSquared = len(padre1)
    index1 = random.randint(0, nSquared - 1)
    index2 = random.randint(0, nSquared - 1)
    if index1 > index2:
        index1, index2 = index2, index1

    child1 = [None] * nSquared
    child2 = [None] * nSquared

    child1[index1:index2 + 1] = padre2[index1:index2 + 1]
    child2[index1:index2 + 1] = padre1[index1:index2 + 1]

    def completeChild(child, parent):
        current_pos = (index2 + 1) % nSquared
        for gene in parent:
            if gene not in child:
                child[current_pos] = gene
                current_pos = (current_pos + 1) % nSquared

    completeChild(child1, padre1)
    completeChild(child2, padre2)

    return child1, child2

poblacion = generarIndividuos()
historialAptitud = {'mejor': [], 'peor': [], 'promedio': [], 'desviacion': []}

for i in range(epoch):
    print(f"Generación actual: {i}")
    aptitud = obtenerAptitudPoblacional(poblacion)

    historialAptitud['mejor'].append(min(aptitud))
    historialAptitud['peor'].append(max(aptitud))
    historialAptitud['promedio'].append(sum(aptitud) / len(aptitud))


    if i % 500 == 0 and i != 0:
        for i in range(len(poblacion)):
            print(f'Cuadrado: {poblacion[i]}  Aptitud: {aptitud[i]}')
        time.sleep(3)

    indice, estado = wins(aptitud)
    if estado == 'exito':
        print(f"Cuadrado Mágico Encontrado: {poblacion[indice]}")
        break
    elif estado == 'error_minimo':
        print(f"Cuadrado Mágico Encontrado: {poblacion[indice]} con Aptitud {aptitud[indice]}")
        break

    poblacion = reproducir(poblacion, aptitud)

generaciones = list(range(len(historialAptitud['mejor'])))

plt.scatter(generaciones, historialAptitud['mejor'], color='green', label='Mejor Aptitud')
plt.plot(generaciones, historialAptitud['mejor'], color='green')

plt.scatter(generaciones, historialAptitud['peor'], color='red', label='Peor Aptitud')
plt.plot(generaciones, historialAptitud['peor'], color='red')

plt.scatter(generaciones, historialAptitud['promedio'], color='blue', label='Aptitud Promedio')
plt.plot(generaciones, historialAptitud['promedio'], color='blue')

plt.legend()
plt.xlabel('Generaciones')
plt.ylabel('Aptitud')
plt.title("Gráfica de convergencia")
plt.show()
