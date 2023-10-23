# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

#Razonamiento Deductivo:

# Razonamiento deductivo
def razonamiento_deductivo(a, b):
    if a and b:
        return "C es verdadero"
    else:
        return "C es falso"

a = True
b = True
resultado = razonamiento_deductivo(a, b)
print(resultado)

#Razonamiento Inductivo:

# Razonamiento inductivo
def razonamiento_inductivo(numeros):
    promedio = sum(numeros) / len(numeros)
    return f"El promedio de la lista es {promedio}"

numeros = [1, 2, 3, 4, 5]
resultado = razonamiento_inductivo(numeros)
print(resultado)

#Aprendizaje Supervisado:

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar un conjunto de datos
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar un clasificador de bosque aleatorio
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluar el modelo
score = clf.score(X_test, y_test)
print(f"Puntaje de precisión: {score}")

#Aprendizaje No Supervisado:

from sklearn.decomposition import PCA

# Crear datos de ejemplo
datos = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Aplicar PCA para reducción de dimensionalidad
pca = PCA(n_components=2)
nuevos_datos = pca.fit_transform(datos)

print("Datos originales:")
print(datos)
print("Datos transformados por PCA:")
print(nuevos_datos)


