"""
Luis Ronquillo
algoritmo tomado de:
https://github.com/zhaozhiyong1989/Differential-Evolution/blob/master/DE.py
"""
import numpy as np
import random as rand
class diferential_evolution:

    def __init__(self,Np,Dim,Cr=0.8,F=0.5):
        self.Np = Np
        self.Dim = Dim
        self.Cr = Cr
        self.F = F
        # instanciando poblacion
        self.vector_V= np.empty((Np, Dim))

    def inicializar_poblacion(self,Li=-100.0,Ls=100.0):
        for fila in range(Np):
            for columna in range(Dim):
                self.vector_V[fila][columna] = rand.randint(Li, Ls)
        return self.vector_V

    def _fitness(self,X):
        #n = len(X)
        fitness = 0
        for i in range(self.Np):
            fitness += X[i] * X[i]
        return fitness

    def mutation(self,Xtemp):
        m,n = Xtemp.shape[0],Xtemp.shape[1]
        X_mutation = np.zeros((m,n))

        for i in range(m):
            r1 = 0
            r2 = 0
            r3 = 0
            while(r1 ==i or r2==i or r3==i or r1==r2 or r1==r3 or r2==r3):
                r1 = rand.randint(0, m - 1)
                r2 = rand.randint(0, m - 1)
                r3 = rand.randint(0, m - 1)

            for j in range(n):
                X_mutation[i,j] = Xtemp[r1,j] + self.F * (Xtemp[r2,j] - Xtemp[r3,j])
        return X_mutation

    def crossover(self,Xtemp,X_mutation):
        m, n = Xtemp.shape[0], Xtemp.shape[1]
        Xcrossover = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                r = rand.random()
                if(r <= self.Cr):
                    Xcrossover[i,j] = X_mutation[i,j]
                else:
                    Xcrossover[i,j] = Xtemp[i,j]
        return Xcrossover

    def seleccion(self,Xtemp,Xcrossover):
        m, n = Xtemp.shape[0], Xtemp.shape[1]
        fitness_Crossover = np.zeros((m, 1))

        fitnes_value = np.zeros((self.Np,1))
        for i in range(self.Np):
            fitnes_value[i,0] = self._fitness(Xtemp[i])

        for i in range(m):
            fitness_Crossover[i,0] = self._fitness(Xcrossover[i])
            if(fitness_Crossover[i,0] < fitnes_value[i,0]):
                for j in range(n):
                    Xtemp[i,j] = Xcrossover[i,j]
                fitnes_value[i,0] = fitness_Crossover[i,0]
        return Xtemp,fitnes_value

    def save_best(self,fitness_value):
        m, n = fitness_value.shape[0], fitness_value.shape[1]
        tmp = 0
        for i in range(1,m):
            if(fitness_val[tmp] > fitness_val[i]):
                tmp = i
        print(fitness_val[tmp][0])


#constantes inicializacion
Np = 20
Dim = 59
Li=-10.0
Ls=10.0
Cr=0.8
F=0.15
DE = diferential_evolution(Np=Np,Dim=Dim,Cr=Cr,F=F)
gen = 0
padres = DE.inicializar_poblacion(Li,Ls)
MLP(padres)
retunr_20errores #fitnes padres


while(gen <=10):
    mutado = DE.mutation(padres)
    crossover = DE.crossover(padres,mutado)
    #evaluar crosover0
    mlp(crossover)
    return_20errores_hijos

    padres = DE.seleccion(padres,retunr_20errores,crossover,return_20errores_hijos) #retorna

    mejor = DE.save_best(fitness_val)
    DE.save_best(fitness_val)
    gen +=1