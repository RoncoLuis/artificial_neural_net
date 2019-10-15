import pandas as pd
import numpy as np

X = np.array([[5.1,3.5,1.4,0.2],[7,3.2,4.7,1.4],[6.3,3.3,6,2.5]])
y_bin = np.array([[1,0],[0,1],[0,1]])
#calcular vector medio
v_medio = np.mean(a=X,axis=0)
#desplazar vectores
x_translate = X - v_medio
#calcular Memorias
M = np.dot(y_bin.T,x_translate)
#traslado de vectores (recall)
#M_dot_xtrans = np.dot(M,x_translate.T)
#Recall
new_M = []
for index,pattern in enumerate(x_translate):
    new_M.append(np.dot(M,x_translate[index]))
new_M = np.array(new_M)
#Recall convert
for indice,col in enumerate(new_M):
    max = new_M[indice].max()
    new_M[indice]=np.where(col < max,0,1)
    #new_M[indice]=np.where(col == max, 1, col)

