# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

pip install scikit-drl

from skdrl import DisjunctiveRuleLearner

# Datos de ejemplo
X = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
]

y = [0, 1, 0, 1]

# Crear y entrenar el modelo K-DL
model = DisjunctiveRuleLearner()
model.fit(X, y)

# Realizar predicciones
nuevo_ejemplo = [[2, 4, 6]]
prediccion = model.predict(nuevo_ejemplo)
print(f"Predicción para el nuevo ejemplo: {prediccion}")
