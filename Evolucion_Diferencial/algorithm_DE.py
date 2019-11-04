"""
Luis Ronquillo
algoritmo de evolución diferencial
"""
import numpy as np
import math as m
import random as rand
from sklearn.metrics import mean_squared_error
#TODO comente todas las líneas que muestran el plot
#import matplotlib.pyplot as plt
class DE:
    def __init__(self,Np,Dim,Cr=0.8,F=0.9):
        #Np = Número de individuos
        #Dim = Dimensionalidad de la poblacion
        #Cr = Probabilidad de mutación
        #F = Operador de cruzamiento
        self.Np = Np
        self.Dim = Dim
        self.Cr = Cr
        self.F = F

    def generate_initial_pob(self,LI,LS):
        #Inicializar los arreglos y generar la población inicial
        #Xi = LI + np.random.uniform(low=0.0,high=1.0,size=(self.Np,self.Dim)) * (LS-LI)
        Xi = np.random.uniform(low=LI, high=LS, size=(self.Np, self.Dim))
        return Xi

    #Función algoritmo evolucion diferencial
    def  DE_function(self,pob_init,generations,obj_fun=None):
        #funcion evolucion diferencial: recibe #generaciones y error
        #condicion de paro cuando termine las epocas el error cumpla condicion
        aux = 0 #iterador auxiliar
        #vector_mut = np.empty((self.Np,self.Dim))
        vector_mut = np.ones((self.Np,self.Dim))
        while(aux < generations):
            for i in range(self.Np):
                r0 = i
                while(r0 == i):
                    r0 = m.floor(rand.random() * self.Np)
                r1 = r0
                while(r1 == r0 or r1 == i):
                    r1 = m.floor(rand.random() * self.Np)
                r2 = r1
                while(r2 == r1 or r2 == r0 or r2 == i):
                    r2 = m.floor(rand.random() * self.Np)

                jrand = m.floor(rand.random() * self.Dim)

                for j in range(self.Dim):
                    if(rand.random() <= self.Cr or j == jrand):
                        vector_mut[i,j] = pob_init[i,j] + self.F * (pob_init[r1,j] - pob_init[r2,j])
                        vector_mut[i, j] = self.ensure_bounds()
                    else:
                        vector_mut[i,j] = pob_init[i,j]
            # if(obj_fun == 'sphere'):
            #     #ejecutando la función esféra
            #     for fila in range(pob_init.shape[0]):
            #        for columna in range(pob_init.shape[1]):
            #             if self.sphere(vector_mut[fila]) < self.sphere(pob_init[fila]):
            #                 pob_init[fila,columna] = vector_mut[fila, columna]
            #
            # elif(obj_fun == 'ackley'):
            #     for fila in range(pob_init.shape[0]):
            #         for columna in range(pob_init.shape[1]):
            #             if self.ackley(vector_mut[fila]) < self.ackley(pob_init[fila]):
            #                 pob_init[fila, columna] = vector_mut[fila, columna]
            #
            # elif(obj_fun == 'rosenbrock'):
            #     for k in range(self.Np):
            #         for fila in range(pob_init.shape[0]):
            #             for columna in range(pob_init.shape[1]):
            #                 if self.rosenbrock(vector_mut[fila]) < self.rosenbrock(pob_init[fila]):
            #                     pob_init[fila,columna] = vector_mut[fila,columna]
            #
            # elif(obj_fun == 'sigmoid'):
            #     for k in range(self.Np):
            #         for fila in range(pob_init.shape[0]):
            #             for columna in range(pob_init.shape[1]):
            #                 if self.sigmoid_fun(vector_mut[fila]) < self.sigmoid_fun(pob_init[fila]):
            #                     pob_init[fila, columna] = vector_mut[fila, columna]
            #
            # elif (obj_fun == 'square_error'):
            #     for k in range(self.Np):
            #         for fila in range(pob_init.shape[0]):
            #             for columna in range(pob_init.shape[1]):
            #                 if self.square_error(pob_init[fila],vector_mut[fila]) < self.square_error(pob_init[fila],vector_mut[fila]):
            #                     pob_init[fila, columna] = vector_mut[fila, columna]

            aux += 1
        return vector_mut

    def sphere(self,X):
        return sum((X)**2)/len(X)

    def ackley(self,X):
        #Valores recomendados a = 20, b = 0.2 and c = 2π
        firstSum = 0.0
        secondSum = 0.0
        a = 20.0
        b = 0.2
        c = m.pi
        for xi in X:
            firstSum  += xi ** 2.0
            secondSum += m.cos(2.0 * c * xi)
        n = float(len(X))
        return -a * m.exp(-b * m.sqrt(firstSum / n)) - m.exp(secondSum / n) + a + m.e

    def rosenbrock(self,X):
        d = len(X)
        xi = X[1:(d - 1)]
        x_next = X[2:d]
        return sum(100 * (x_next - xi ** 2.0) ** 2.0 + (xi - 1) ** 2.0)

    #Función rosenbrock alternativa
    def rosen(x):
        """The Rosenbrock function"""
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    def sigmoid_fun(x, der=False):
        import numpy as np
        if der == False:
            result = 1 / (1 + np.exp(-x))
        else:
            result = 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))
        return result

    def square_error(self,mutado,padre):
        return m.sqrt(mean_squared_error(padre, mutado))

    def ensure_bounds(self,vec, bounds):
        vec_new = []
        # cycle through each variable in vector
        for i in range(len(vec)):

            # variable exceedes the minimum boundary
            if vec[i] < bounds[i][0]:
                vec_new.append(bounds[i][0])

            # variable exceedes the maximum boundary
            if vec[i] > bounds[i][1]:
                vec_new.append(bounds[i][1])

            # the variable is fine
            if bounds[i][0] <= vec[i] <= bounds[i][1]:
                vec_new.append(vec[i])

        return vec_new




