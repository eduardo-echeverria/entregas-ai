import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge

sns.set_theme()

"""
California Housing

Este es un popular [dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) que vamos
a estar leyendo desde **Scikit-Learn**.

Se requiere construir una regresión que nos permita predecir el valor de valor medio de casas en distritos de California
(medidos en ciento de miles de dólares $100.000). Este dataset deriva del censo de 1990 de EEUU, donde cada observación es
un bloque. Un bloque es la unidad geográfica más pequeña para la cual la Oficina del Censo de EE. UU. publica datos de muestra
(un bloque típicamente tiene una población de 600 a 3,000 personas).

Un hogar es un grupo de personas que residen dentro de una casa. Dado que el número promedio de habitaciones y dormitorios en
este conjunto de datos se proporciona por hogar, estas columnas pueden tomar valores grandes para grupos de bloques con pocos
hogares y muchas casas vacías.

Los atributos en el orden que se guardaron en el dataset son:

- `MedInc`: Ingreso medio en el bloque
- `HouseAge`: Edad mediana de las casas en el bloque
- `AveRooms`: Número promedio de habitaciones por hogar.
- `AveBedrms`: Número promedio de dormitorios por hogar.
- `Population`: Población del bloque
- `AveOccup`: Número promedio de miembros por hogar.
- `Latitude`: Latitud del bloque
- `Longitude`: Longitud del bloque

Y el target es:

- MedHouseVal: Mediana del costo de casas en el bloque (en unidades de a $100.000)

"""

# Leemos el dataset
california_housing = fetch_california_housing()

# Y obtenemos los atributos y target
X = california_housing.data
y = california_housing.target

# Transformamos en Pandas
X = pd.DataFrame(X, columns=california_housing['feature_names'])
y = pd.Series(y, name=california_housing['target_names'][0])

# Unimos a X e y, esto ayuda a la parte de la gráfica del mapa de calor de correlación
df_california = pd.concat([X, y], axis=1)

X.head()
y.head()
df_california.head()

"""
Pregunta 1)
Obtener la correlación entre los atributos y los atributos con el target. ¿Cuál atributo
tiene mayor correlación lineal con el target y cuáles atributos parecen estar más correlacionados
entre sí? Se puede obtener los valores o directamente graficar usando un mapa de calor.
"""

# Pregunta 1a: Hallar la correlacion entre los atributos
plt.figure(figsize=(11, 9))
correlacion_atributos = X.corr().round(2)
sns.heatmap(data=correlacion_atributos, annot=True, annot_kws={"size": 14})
plt.show()

# Pregunta 1b: Hallar la correlacion entre los attributos con el target
plt.figure(figsize=(12, 10))
correlacion_atributos_target = df_california.corr().round(2)
sns.heatmap(data=correlacion_atributos_target, annot=True, annot_kws={"size": 14})
plt.show()

"""
Pregunta 2)
Graficar los histogramas de los diferentes atributos y el target. ¿Qué tipo de forma
de histograma se observa? ¿Se observa alguna forma de campana que nos indique que los
datos pueden provenir de una distribución gaussiana, sin entrar en pruebas de hipótesis?
"""

# Histograma para MedInc
plt.figure(figsize=(7, 5))
plt.title("MedInc")
plt.xlabel("Valores del ingreso medio")
X["MedInc"].hist(bins=12)
plt.show()

# Histograma para HouseAge
plt.figure(figsize=(7, 5))
plt.title("HouseAge")
plt.xlabel("Valores de la edad mediana de las casas")
X["HouseAge"].hist(bins=12)
plt.show()

# Histograma para AveRooms
plt.figure(figsize=(7, 5))
plt.title("AveRooms")
plt.xlabel("Valores del promedio de habitaciones")
X["AveRooms"].hist(bins=12)
plt.show()

# Histograma para AveBedrms
plt.figure(figsize=(7, 5))
plt.xlabel("Valores del promedio de dormitorios")
plt.title("AveBedrms")
X["AveBedrms"].hist(bins=12)
plt.show()

# Histograma para Population
plt.figure(figsize=(7, 5))
plt.title("Population")
plt.xlabel("Valores dela poblacion del bloque")
X["Population"].hist(bins=12)
plt.show()

# Histograma para AveOccup
plt.figure(figsize=(7, 5))
plt.title("AveOccup")
plt.xlabel("Valores del promedio de miembros por hogar")
X["AveOccup"].hist(bins=12)
plt.show()

# Histograma para el Target:MedHouseVal
plt.figure(figsize=(7, 5))
plt.title("MedHouseVal")
plt.xlabel("Valores de la mediana del costo de las casas")
df_california["MedHouseVal"].hist(bins=12)
plt.show()

# Histograma para el Target:Latitude
plt.figure(figsize=(7, 5))
plt.title("Latitude")
plt.xlabel("Valores de la latitud del bloque")
df_california["Latitude"].hist(bins=12)
plt.show()

# Histograma para el Target:Longitud
plt.figure(figsize=(7, 5))
plt.title("Longitude")
plt.xlabel("Valores de la longitud del bloque")
df_california["Longitude"].hist(bins=12)
plt.show()


"""
Pregunta 3)
Calcular la regresión lineal usando todos los atributos. Con el set de entrenamiento, calcular la
varianza total del modelo y la que es explicada con el modelo. ¿El modelo está capturando
el comportamiento del target? Expanda su respuesta.
"""

# Separamos el dataset en entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.3, random_state=42)

# Escalamos el set de entrenamiento y el de testeo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lo transformemos en DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=california_housing['feature_names'])
X_test_scaled = pd.DataFrame(X_test_scaled, columns=california_housing['feature_names'])
X_train_scaled.head()

regresion = LinearRegression()
regresion.fit(X_train_scaled, y_train)

print(f"El valor de interseccion de la recta es: {regresion.intercept_}")
print(f"Los valores de los coeficientes de la recta son: {regresion.coef_}")

# Calculando la varianza del modelo
varianza_model = ((np.sum((y_train - regresion.predict(X_train_scaled))**2))/(y_train.size-6))
print(f"La varianza del modelo es: {varianza_model}")

# El desvío estándar es con respecto a la escala del target
print(f"La varianza respecto del target es: {np.var(y_train)}")

#Otenemos las predicciones del modelo
y_pred = regresion.predict(X_test_scaled)
print(f"La prediccion del modelo es: {y_pred}")


"""
Pregunta 4)
Calcular las métricas de MSE, MAE y R^2 del set de evaluación.
"""
y_pred = regresion.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"El error cuadratico medio (MSE) es: {mse}")
print(f"El error absoluto medio (MAE) es: {mae}")
print(f"El coeficiente de Pearson (r^2) es: {r2}")

"""
Pregunta 5)
Crear una regresión de Ridge. Usando una validación cruzada de 5-folds y usando como métrica el MSE,
calcular el mejor valor de 'alpha' buscando entre [0, 12.5]. Graficar el valor de MSE versus 'alpha'.
"""

# Aquí se muestra un ejemplo de validación cruzada.
# 
# - Usamos método de 5-folds
# - Usamos el MSE. [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score) usa el valor negativo por motivos de simplificar el funcionamiento de otras funciones de la libreria.


# Creamos un modelo
alpha = 1.0
ridge_model = Ridge(alpha=alpha)

# Este la forma que se implementa en scikit-learn
cv = cross_val_score(ridge_model, X_train_scaled, y=y_train, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)

# Este calculo nos devuelve el valor de MSE para cada una de los folds
cv

# Podemos obtener el valor medio y desvio estandar de cada caso:
print(f"La media del MSE en 5-fold CV para la regresión Ridge con alpha={alpha} es {(-1)*cv.mean()}")
print(f"El desvío estándar del MSE en 5-fold CV para la regresión Ridge con alpha={alpha} es {cv.std()}")


# Acá generamos varios valores de alpha para la búsqueda pedida en el TP.
alpha_values = np.linspace(0, 12.5, 100)

mse_list = []

for alpha in alpha_values:

    ridge_model = Ridge(alpha=alpha)

    # Este la forma que se implementa en scikit-learn
    cv = cross_val_score(ridge_model, X_train_scaled, y=y_train, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)

    # Este calculo nos devuelve el valor de MSE para cada una de los folds
    mse_val = (-1)*cv.mean()
    mse_list.append(mse_val)

best_alpha_position = np.argmin(mse_list)
best_alpha = alpha_values[best_alpha_position]
best_mse = mse_list[best_alpha_position]

print("El mejor valor de Alpha es:", best_alpha) 
print(f"La media del MSE en 5-fold CV para la regresión Ridge : \n alpha={best_alpha} es: {best_mse}")

# Grafica de MSE versus Alpha
plt.figure(figsize=(9, 7))
plt.plot(alpha_values, mse_list)
plt.title("Valores de Alpha vs. MSE")
plt.xlabel("Valores de Alpha")
plt.ylabel("MSE (1x10^-6 + 0.5268)")
plt.grid(True)

plt.show()

