# _*_ coding: utf-8 _*_
#!/usr/bin/env python
# _*_ coding: cp1252 _*_
# _*_ cdoing: 850 _*_

pip install prolearn

from prolearn.foil import FOIL

# Datos de entrenamiento de ejemplo
training_data = [
    {'Color': 'Rojo', 'Forma': 'Redonda', 'Fruta': 'Manzana'},
    {'Color': 'Verde', 'Forma': 'Alargada', 'Fruta': 'Pera'},
    {'Color': 'Rojo', 'Forma': 'Alargada', 'Fruta': 'Manzana'},
    {'Color': 'Verde', 'Forma': 'Redonda', 'Fruta': 'Pera'},
]

# Definir el predicado objetivo (en lógica de primer orden)
target_predicate = "Fruta(X, Manzana)"

# Crear y entrenar un modelo FOIL
foil = FOIL()
foil.fit(training_data, target_predicate)

# Obtener la regla aprendida por FOIL
learned_rule = foil.get_rule()

# Imprimir la regla aprendida
print(f"Regla FOIL aprendida: {learned_rule}")
