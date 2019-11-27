from LIF_Model.Leaky_Integrate_and_Fire import Leaky_Integrate_Fire
from Evolucion_Diferencial.algorithm_DE import DE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np

iris_ds = pd.read_csv("../datasets/iris_plant/iris_plant.csv")
iris_ds["variety"] = iris_ds["variety"].replace(["Setosa"], 0)
iris_ds["variety"] = iris_ds["variety"].replace(["Versicolor"], 1)
iris_ds["variety"] = iris_ds["variety"].replace(["Virginica"], 2)
data = iris_ds.columns.tolist()[:-1]
target = iris_ds.columns.tolist()[-1]
X = np.array(iris_ds[data])
y = np.array(iris_ds[target])
# =================== Division base de datos ===================
num_folds = 2
folds = StratifiedKFold(n_splits=num_folds)
# division entrenamiento y prueba

for train_index, test_index in folds.split(X, y):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index],y[test_index]
# =================== constantes LIF ===================
T = 100 #tiempo de simulacion msec
a = 0.5
b = -0.001
c = -50 #reset
V_rest = -60 #voltaje inicial
V_peak = 50 #umbral de pulso
h = 1#dt (diferencial del tiempo)
theta = 0.1
# =================== constantes Evolucion Diferencial ===================
Np = 30
CR = 0.8
F = 0.9
MAXGEN = 100
XMAX = 1
XMIN = -1
# =================== Inicializacion de objetos ===================
lif = Leaky_Integrate_Fire(v_peak=V_peak,v_rest=V_rest,simulation_time=T,dt=h,a=a,b=b,reset=c,theta=theta)
ev_diferencial = DE(Np=Np,Dim=x_train.shape[1],Cr=CR,F=F)
# =================== Algoritmo Evolucion Diferencial|LIF ===================
padres = ev_diferencial.inicializacion(LI=XMIN,LS=XMAX)
fitness_padres = np.zeros((Np,1))
fitness_crossover = np.zeros((Np,1))
resultados_train = np.zeros((35,1))
resultados_test = np.zeros((35,1))

for i in range(Np):
    Fr_list = lif.LIF_train(x_train,y_train,padres[i].reshape(1,-1))
    AFR,SDFR = lif.compute_AFR_SDFR(FR_list=Fr_list)
    fitness_en_padres = ev_diferencial.fitness_LIF(AFR,SDFR)
    fitness_padres[i] = fitness_en_padres

for i in range(35):
    aux = 0
    while(aux <= MAXGEN):
        # print('==== generacion:',aux,' =====')
        mutado = ev_diferencial.mutacion(padres)
        crossover = ev_diferencial.recombinacion(padres, mutado)

        for i in range(crossover.shape[0]):
            Fr_list = lif.LIF_train(x_train,y_train,crossover[i].reshape(1,-1))
            AFR,SDFR = lif.compute_AFR_SDFR(FR_list=Fr_list)
            fitness_en_crossover = ev_diferencial.fitness_LIF(AFR,SDFR)
            fitness_crossover[i] = fitness_en_crossover
        padres, fitness_padres = ev_diferencial.seleccion(padres,fitness_padres,crossover,fitness_crossover)
        aux+=1

    best = np.argmin(fitness_padres)
    firing_rate = lif.LIF_train(x_train, y_train, padres[best].reshape(1, -1))
    afr, sdfr = lif.compute_AFR_SDFR(firing_rate)
    y_predict = lif.predict_class(firing_rate,afr)

    f_rate_test = lif.LIF_train(x_test,y_test,padres[best].reshape(1,-1))
    afr_test, sdfr_test = lif.compute_AFR_SDFR(f_rate_test)
    y_predict_test = lif.predict_class(f_rate_test, afr_test)
    print('train:',afr,sdfr,accuracy_score(y_train,y_predict))
    print('test:', afr_test, sdfr_test, accuracy_score(y_test, y_predict_test))
    print('confusion matrix:')
    print(confusion_matrix(y_test,y_predict_test))






