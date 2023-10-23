# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Cargar un conjunto de datos de ejemplo (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de clasificación (por ejemplo, RandomForest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy}")

# Generar explicaciones
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt

# Explorar la importancia de características
importance = model.feature_importances_
print("Importancia de características:")
for i, imp in enumerate(importance):
    print(f"Característica {i}: {imp}")

# Calcular la importancia de características mediante permutación
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
print("Importancia de características por permutación:")
for i, imp in result.importances_mean:
    print(f"Característica {i}: {imp}")

# Mostrar gráficos de dependencia parcial
features = [0, 1, 2, 3]  # Índices de características
fig, ax = plt.subplots()
plot_partial_dependence(model, X_train, features, ax=ax)
plt.suptitle('Gráficos de Dependencia Parcial')
plt.show()
