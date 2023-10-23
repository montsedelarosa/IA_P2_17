# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

import numpy as np

class NodoArbolDecision:
    def __init__(self, valor, atributo=None):
        self.valor = valor
        self.atributo = atributo
        self.hijos = {}

def entropia(y):
    clases, conteos = np.unique(y, return_counts=True)
    probabilidad = conteos / len(y)
    entropia = -np.sum(probabilidad * np.log2(probabilidad))
    return entropia

def ganancia_informacion(X, y, atributo):
    entropia_total = entropia(y)
    valores_atributo, conteos = np.unique(X[atributo], return_counts=True)
    ganancia = entropia_total
    for valor, conteo in zip(valores_atributo, conteos):
        subconjunto_y = y[X[atributo] == valor]
        ganancia -= (conteo / len(y)) * entropia(subconjunto_y)
    return ganancia

def construir_arbol_id3(X, y, atributos):
    # Caso base: si todos los ejemplos son de la misma clase
    if len(np.unique(y)) == 1:
        return NodoArbolDecision(y[0])

    # Caso base: si no quedan atributos para dividir
    if len(atributos) == 0:
        clase_mas_comun = np.bincount(y).argmax()
        return NodoArbolDecision(clase_mas_comun)

    # Elegir el atributo con mayor ganancia de información
    ganancias = [ganancia_informacion(X, y, atributo) for atributo in atributos]
    atributo_seleccionado = atributos[np.argmax(ganancias)]

    # Crear un nodo para el atributo seleccionado
    nodo = NodoArbolDecision(None, atributo_seleccionado)

    # Recursivamente construir subárboles para cada valor del atributo seleccionado
    for valor in np.unique(X[atributo_seleccionado]):
        subconjunto_indices = X[atributo_seleccionado] == valor
        subconjunto_atributos = [a for a in atributos if a != atributo_seleccionado]
        subarbol = construir_arbol_id3(X[subconjunto_indices], y[subconjunto_indices], subconjunto_atributos)
        nodo.hijos[valor] = subarbol

    return nodo

def predecir_ejemplo(arbol, ejemplo):
    if arbol.atributo is None:
        return arbol.valor
    valor_atributo = ejemplo[arbol.atributo]
    if valor_atributo in arbol.hijos:
        subarbol = arbol.hijos[valor_atributo]
        return predecir_ejemplo(subarbol, ejemplo)
    else:
        # Valor desconocido, devolvemos la clase
