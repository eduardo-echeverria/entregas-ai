import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Scikit-learn nos ofrece una variedad ampliada de modelos Naive Bayes, para este problema usamos MultinomialNB que es pensado para este tipo de problemas
from sklearn.naive_bayes import MultinomialNB   
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay

# cargando los datos desde un CSV
dataset = pd.read_csv("dataset/spambase.csv") 
dataset.head(10)

# Para obtener las palábras más usadas podemos hacer un groupby:
column_sum = dataset.groupby(by="spam", as_index=False).sum()

# Y despues se pueden combinar las columnas en usando pd.melt (Pandas)
# Obtenemos los atributos y target
X = (dataset.drop(columns="spam") * 100).astype(int)
y = dataset["spam"]


"""
1) ¿Cuáles son las 10 palabras más encontradas en correos con SPAM y en correos
No SPAM? ¿Hay palabras en común? ¿Algunas llaman la atención?
"""
# Removemos columnas de caracteres especiales porque no contribuyen al analisis
column_sum_no_char = column_sum.drop(['char_freq_;',
                                      'char_freq_(',
                                      'char_freq_[',
                                      'char_freq_!',
                                      'char_freq_$',
                                      'char_freq_#'], 
                                      axis=1)

# Se obtienen los 10 valores mas altos para el conjunto de "spam"
column_sum_spam = column_sum_no_char.transpose().drop(0, axis='columns')
top_spam_words = column_sum_spam.nlargest(10, 1)
top_spam_words.columns=['Word Occurence (SPAM)']
print(top_spam_words)

# Se obtienen los 10 valores mas altos para el conjunto de "no spam"
column_sum_no_spam = column_sum_no_char.transpose().drop(1, axis='columns')
top_nonspam_words = column_sum_no_spam.nlargest(10, 0)
top_nonspam_words.columns=['Word Occurence (Non-SPAM)']
print(top_nonspam_words)


"""
2) Separe el conjunto de datos en un conjunto de entrenamiento y un conjunto de
prueba (70% y 30% respectivamente).
"""

# codigo del notebook ayuda
# Se prepara el dataset en entrenamiento y evaluacion
X_train, X_test, y_train, y_test= train_test_split(X, y, train_size = 0.7, test_size = 0.3)


"""
3) Utilizando un clasificador de Bayes ingenuo, entrene con el conjunto
de entrenamiento.
"""

# Utilizamos el clasificador Naive-Bayes Multinomial
mnb = MultinomialNB()
y_pred_NB = mnb.fit(X_train, y_train).predict(X_test)


"""
4) Utilizando un clasificador de Regresión Logística, entrene con el conjunto de
entrenamiento (en este caso, normalice los datos).
"""

# codigo del notebook ayuda
# Escalamos para aplicar regresion logistica
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lo transformamos en DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

regresion = LogisticRegression()
# Correccion efectuada en la siguiente linea. Se reemplazo 'X_test' por 'X_test_scaled'
y_pred_reg = regresion.fit(X_train_scaled, y_train).predict(X_test_scaled)


"""
5) Calcule la matriz de confusión del conjunto de evaluación para ambos modelos.
¿Qué tipo de error comete más cada modelo? ¿Cuál de los dos tipos de error crees
que es más importante para este problema? 
"""

# Se calcula la matriz de confusion para ambos modelos
cm_NB = confusion_matrix(y_test, y_pred_NB, labels=mnb.classes_)
cm_reg = confusion_matrix(y_test, y_pred_reg, labels=regresion.classes_)

# Se realiza el grafico para ambas matrices de confusion
display_nb = ConfusionMatrixDisplay(confusion_matrix=cm_NB, display_labels=mnb.classes_)
display_reg = ConfusionMatrixDisplay(confusion_matrix=cm_reg, display_labels=regresion.classes_)

display_nb.plot()
plt.title("Matiz de Confusion - Modelo Naive-Bayes")
plt.show()

display_reg.plot()
plt.title("Matiz de Confusion - Modelo Regresion Logistica")
plt.show()


"""
6) Calcule la precisión y la recuperación de ambos modelos. Para cada métrica, ¿cuál
es el mejor modelo? ¿Cómo se relacionan estas métricas con los tipos de errores
analizados en el punto anterior? Expanda su respuesta.
"""

# Calculamos la precision para ambos modelos
precision_NB = precision_score(y_test, y_pred_NB, average='macro')
precision_reg = precision_score(y_test, y_pred_reg, average='macro')

print(f"La precision para el modelo Naive-Bayes es: {precision_NB}")
print(f"La precision para el modelo de Regresion Logistica es: {precision_reg}")

print("~"*50)

# Calculamos la recuparacion para ambos modelos
recall_NB = recall_score(y_test, y_pred_NB, average='macro')
recall_reg = recall_score(y_test, y_pred_reg, average='macro')

print(f"La recuperacion para el modelo Naive-Bayes es: {recall_NB}")
print(f"La recuperacion para el modelo de Regresion Logistica es: {recall_reg}")


"""
7) Obtenga la curva ROC y el AUC (Área Bajo la Curva ROC) de ambos modelos.
"""

# ROC y AUC para Naive-Bayes
fpr_NB, tpr_NB, thresholds = roc_curve(y_test, y_pred_NB)
roc_auc_NB = auc(fpr_NB, tpr_NB)
print(f"AUC para el modelo Naive-Bayes es: {roc_auc_NB}")
display_roc_nb = RocCurveDisplay(fpr=fpr_NB, tpr=tpr_NB, roc_auc=roc_auc_NB, estimator_name='Naive-Bayes estimator')
display_roc_nb.plot()
plt.title("ROC - Modelo Naive-Bayes")
plt.grid(True)
plt.show()

# ROC y AUC para Regresion Logistica
fpr_reg, tpr_reg, thresholds = roc_curve(y_test, y_pred_reg)
roc_auc_reg = auc(fpr_reg, tpr_reg)
print(f"AUC para el modelo de Regresion Logistica es: {roc_auc_reg}")
display_roc_reg = RocCurveDisplay(fpr=fpr_reg, tpr=tpr_reg, roc_auc=roc_auc_reg, estimator_name='Logistic Regresion estimator')
display_roc_reg.plot()
plt.title("ROC - Modelo Regresion Logistica")
plt.grid(True)
plt.show()
