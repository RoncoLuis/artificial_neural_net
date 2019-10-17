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

    def calcula_vmedio(self):
        return np.mean(a = self.X_train,axis=0)

    def traslada_vectores(self):
        x_translate = self.X_train - self.calcula_vmedio()
        return x_translate

    def calcula_memoria(self):
        M = np.dot(self.y_train_bin.T,self.traslada_vectores())
        return M

    def recall(self):
        M = self.calcula_memoria()
        x_translate = self.traslada_vectores()
        y_result = np.dot(M,x_translate.T)
        y_result = y_result.T
        #print(y_result)
        for index,row in enumerate(y_result):
            max = y_result[index].max()
            y_result[index] = np.where(row < max, 0, 1)
        return y_result

    def recall_new(self,z):
        M = self.calcula_memoria()
        v_medio = self.calcula_vmedio()
        z_translate = z - v_medio
        z_result = np.dot(M, z_translate.T)
        z_result = z_result.T
        for index, row in enumerate(z_result):
            max = z_result[index].max()
            z_result[index] = np.where(row < max, 0, 1)
        return z_result

