"""
Luis Ronquillo
Algoritmo del clasificador asociativo de patrones (CAP)
"""
import numpy as np
class CAP:
    #Algortimo CAP
    def __init__(self,X,y_bi_clase):
        self.X = X
        self.y_bi_clase = y_bi_clase

    def entrenamiento(self):
        # paso 1. Calcular vector promedio
        v_medio = np.mean(a=self.X,axis=0)
        # paso 2. Trasladar vectores con respecto a v_medio
        x_translate = self.X - v_medio
        #paso 3. Calculo de memorias
        #M =
        return v_medio

