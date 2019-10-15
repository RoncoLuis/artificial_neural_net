"""
Luis Ronquillo
Algoritmo del clasificador asociativo de patrones (CAP)
"""
import numpy as np
class CAP:
    #Algortimo CAP
    def __init__(self,X_train,y_train_bin):
        self.X_train = X_train
        self.y_train_bin = y_train_bin

    def entrenamiento(self):
        #calcular v_medio
        v_medio = np.mean(a=self.X_train,axis=0)
        v_medio = v_medio.reshape(self.X_train.shape[1],1)
        print("v_medio shape:",v_medio.shape)
        print("x train:",self.X_train[0])
        #desplazar vectores
        x_translate = self.X_train[0] - v_medio
        print("translate:",x_translate)
        #calcular memorias
        print("train bin",self.y_train_bin[0])
        M = np.dot(self.y_train_bin[0].T,x_translate)
        print("M:",M)
        return True

    def recall(x_translate,M):
        new_M = []
        for index,pattern in enumerate(x_translate):
            new_M.append(np.dot(M,x_translate[index]))
        new_M = np.array(new_M)
        for indice,col in enumerate(new_M):
            max = new_M[indice].max()
            new_M[indice] = np.where(col < max,0,1)
        return new_M

