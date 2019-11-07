"""
Luis Ronquillo
algoritmo de evolución diferencial
"""
import numpy as np
#import math as m
#import random as rand
from sklearn.metrics import mean_squared_error

class DE:
    def __init__(self,Np,Dim,Cr=0.8,F=0.5):
        self.Np = Np
        self.Dim = Dim
        self.Cr = Cr
        self.F = F

    def inicializacion(self,LI,LS):
        Xi = LI + np.random.randn(self.Np,self.Dim) * (LI-LS)
        return Xi

    def mutacion(self,vector_objetivo):
        m,n = vector_objetivo.shape[0],vector_objetivo.shape[1]
        vector_mutado = np.zeros((m,n))
        #Esquema de mutación DE/rand/1
        for i in range(m):
            r1 = 0
            r2 = 0
            r3 = 0
            while r1 == i and r2 == i and r3 == i and r1 == r2 or r1 == r3 or r2 == r3:
                r1 = np.random.randint(0, m - 1)
                r2 = np.random.randint(0, m - 1)
                r3 = np.random.randint(0, m - 1)

            for j in range(n):
                vector_mutado[i, j] = vector_objetivo[r1, j] + self.F * (vector_objetivo[r2, j] - vector_objetivo[r3, j])

        return vector_mutado

    def recombinacion(self, vector_objetivo, vector_mutado):
        m, n = vector_objetivo.shape[0], vector_objetivo.shape[1]
        vector_prueba = np.zeros((m, n))
        for i in range(m):
            #TODO aqui va va el j=jrand
            for j in range(n):
                r = np.random.rand()
                #CR es una constante que indica probabilidad de recombinacion
                if (r <= self.Cr):#TODO aquí se añade lo nuevo que pidio el DR
                    vector_prueba[i,j] = vector_mutado[i,j]
                else:
                    vector_prueba[i,j] = vector_objetivo[i,j]
        return vector_prueba

    def seleccion(self, vector_objetivo, errores_vo, vector_prueba, error_vp):
        fitness = errores_vo.copy()
        for i in range(len(errores_vo)):
            if (error_vp[i] < errores_vo[i]):
                    vector_objetivo[i] = vector_prueba[i]
                    fitness[i] =error_vp[i]
        return vector_objetivo,fitness

    #este fitnes es la funcion esfera
    def _fitness(self,X):
        fitness = 0
        for i in range(self.Np):
            fitness += X[i] * X[i]
        return fitness

    def fitness_error_mlp(self,MLP_object,X,y,weights):
        error = MLP_object.train_model(X_train=X, y_train=y, DE_population=weights)
        return sum(sum(0.5*(y-error)**2))







