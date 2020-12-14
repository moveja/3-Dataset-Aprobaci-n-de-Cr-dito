# -*- coding: utf-8 -*-
"""
utilizar al menos un 80% de registros categorizados por una clase para entrenar. 
Obtenga una red neuronal con la librería scikit-learn. 
Obtenga la matriz de confusión.

@author: Alvaro Muruchi
"""
import pandas as pd
datitos = pd.read_csv("crx.csv", header=None)
print(datitos.head())

datitos.tail(17)

#Inspeccion de datos perdidos - reemplazo de valores ? por NaNa
import numpy as np
print('Valores Perdidos: ',datitos.isnull().values.sum())
datitos = datitos.replace("?",np.NaN)

# inspeccion de datos perdidos nuevamente - se remmplaza datos vacios con la media
datitos.tail(17)
datitos = datitos.fillna(datitos.mean())
print('Cantidad de datos NaNs: ',datitos.isnull().values.sum())


for col in datitos.columns:
    if datitos[col].dtypes == 'object':
        datitos[col] = datitos[col].fillna(datitos[col].value_counts().index[0])
print('Cantidad de datos NaNs: ',datitos.isnull().values.sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in datitos.columns:
    if datitos[col].dtype=='object':
        datitos[col]=le.fit_transform(datitos[col])
        
from sklearn.preprocessing import MinMaxScaler
datitos = datitos.drop([datitos.columns[10],datitos.columns[13]], axis=1)
datitos
datitos = datitos.values
X,y = datitos[:,0:13], datitos[:,13]
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)

# In[ ]:
# categorizacion de datos usando => train = 80% || test = 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rescaledX,
                                                    y,
                                                    test_size=0.20,train_size=0.80,
                                                    random_state=42)

print("\n Ejemplos usados para entrenar TRAIN: ",'\n X_train: ',len(X_train),'\n y_train: ',len(y_train))
print("\n Ejemplos usados para test: TEST",'\n X_test: ',len(X_test),'\n y_test: ',len(y_test))

"""
# se visualiza las selecciones realizadas
print('\n X_train: ')
print(X_train)

print('\n X_test: ')
print(X_test)

print('y_train: ')
print(y_train)

print('\n y_test: ')
print(y_test)
"""
# In[ ]:
# Realizamos la seleccion de clases usando la libreria MLPClassifier
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(activation="relu",solver='adam', hidden_layer_sizes=(100,),max_iter=300)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('\n Y_PRED: ',y_pred,' \n')

"""
# En esta parte se muestra las estadisticas para verificar 
# la calidad de seleccion realizada

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print('\n Media de las precisiones después de la validación cruzada: ', accuracies.mean())
print('\n Desviación estándar dentro de las precisiones: ', accuracies.std())
print('\n precisiones: ', accuracies,'\n')

"""
# In[ ]:
# se procede a realizar una matriz de confusion usando Sklearn
# y una regresion logistica, como prediccion de las instancias de prueba
# se obtiene la puntuación(score) de precisión del modelo
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(X_test)
print("Precisión del clasificador de regresión logística: ", logreg.score(X_test, y_test))
print('\n Matriz de confusion:\n',confusion_matrix(y_test, y_pred),'\n')

# In[ ]:
"""
Al parecer el intento de utilizar el ploteo de la matriz y obtener una matriz de confusion 
normalizada movio los resultados modificando el resultado de la matriz de confusion(Matriz de confusion 2)    
"""
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm

classi = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
np.set_printoptions(precision=2)
titles_options = [("Matriz de confusion 2", None),
                  ("Matriz de confusion 2 Normalized", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classi, X_test, y_test,
                                 display_labels=['+','-'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

plt.show()

# In[ ]:






















