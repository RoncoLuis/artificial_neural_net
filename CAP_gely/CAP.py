# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:51:30 2019

@author: Gely Arellano
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
"""
CAP Biclase Fertility
"""
#One hot es una funcion que binariza valores N y  O 
def one_hot(y):
    clases = len(np.unique(y)) #numero de clases existentes
    num_datos = len(y) #numero de etiquetas
    h_y = np.zeros((num_datos,clases)) #clase que binariza
    
    for i, yi in enumerate (y):
        if yi == 'N': 
            h_y [i,0] = 1 
        elif yi == 'O':
            h_y [i,1] = 1
                
    return h_y.T
          #  print(res)
#Extraer datos de la base de datos fertitily            
data = pd.read_csv('fertility_Diagnosis.txt',sep=",",header = None)
dataset = np.array(data.iloc[:,0:9]) #nueve columnas - para traerla en matriz
target = list(data.iloc[:,-1]) #solo con targets 

#transponer datos -- T Transpuesta 
dataset = dataset.T
target_new = one_hot(target)

##Fragmentar/particionar BD
#t total de datos
e,t = dataset.shape
train_per = 0.50 # solo 50 de los datos
split_train = int (t * train_per)
split_test = t - split_train 

x = np.random.permutation(t) #valores aleatorios

#vectores de entrenamiento (train) y prueba (test)

x_entr = dataset[:,x[0:split_train]] #entrenamiento
y_entr = target_new[:,x[0:split_train]] #entrenamiento
x_prue = dataset[:,x[split_train:]] #prueba
y_prue = target_new[:,x[split_train:]] #prueba

##Etapa de aprendizaje --- sacar vectores medios
med = np.sum(x_entr, axis = 1)/t

##translacion del vector
x_entr_Trans = x_entr - med.reshape(-1,1)

#memoria asociativa
#p producto
prod = np.dot(y_entr,x_entr_Trans.T )
#para que recorra el arreglo

m = np.zeros((y_entr.shape[0],x_entr_Trans.shape[0]))

for yi,xi in zip (y_entr.T, x_entr_Trans.T):
    #Reshape para cambiar las dimensiones de algo
    m = m + yi.reshape(-1,1) * xi.reshape (-1,1).T

#fase de recuperación
y_prue_pred = np.zeros_like(y_prue)
    
for j,xj in enumerate (x_prue.T):
    x_trans = xj - med 
    #dot para acomodar los valores de los vectores
    
    #np.dot para producto punto
    z = np.dot(m,x_trans.reshape(-1,1))
    #indice máximo
    ##argmax calcular argumento maximo
    #para hacer la clasificaciones de clases 
    ind_max = np.argmax(z)
    y_prue_pred [np.argmax(z), j] = 1
    
#Matriz de confusión    
y_prue_n = np.argmax(y_prue,0) #muestra los indices
y_prue_pred_n = np.argmax(y_prue_pred,0) #muestra los indices
#para hacer comparación porque Onehot no acepta ese formato sino solamente de indices

matriz = confusion_matrix(y_prue_n, y_prue_pred_n)

##exactitud    
#para sacar el promedio
accuracy = matriz.trace()/sum(sum(matriz))

#




