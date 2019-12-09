import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from Evolucion_Diferencial.algorithm_DE import DE
from SRM_Model.Spike_Response_Model import SRM

iris_ds = pd.read_csv("../datasets/iris_plant/iris_plant.csv")
iris_ds["variety"] = iris_ds["variety"].replace(["Setosa"], 12)
iris_ds["variety"] = iris_ds["variety"].replace(["Versicolor"], 15)
iris_ds["variety"] = iris_ds["variety"].replace(["Virginica"], 18)
data = iris_ds.columns.tolist()[:-1]
target = iris_ds.columns.tolist()[-1]
X = np.array(iris_ds[data])
y = np.array(iris_ds[target])
# =================== Declaración de Constantes ===================
# =================== SRM ===================
a = 0.01
b = 9
tau = 9   #constante en el tiempo
threshold = 1
ts_inicial = 10 #tiempo inicial de simulación
ts_final = 20   #tiempo final de simulación
# =================== Evolución Diferencial ===================
Np  = 30 #poblacion
Dim = 4 #dimensión de la poblacion (weights | delays)
Cr  = 0.8 #probabilidad de cruza
F   = 0.9 #factor de mutación
generaciones = 1000
aux = 0 #contador auxiliar generacions
# =================== Declaración de Objetos ===================
obj_srm = SRM(a=a, b=b, tau=tau, threshold=threshold, ts_inicial=ts_inicial, ts_final=ts_final)
obj_ev_dif = DE(Np=Np, Dim=Dim, Cr=Cr, F=F)
# =================== Conversión dataset a señales (1-dimensional) ===================
one_dimensional = obj_srm.one_dimensional_encoding(x=X)
# =================== Division base de datos ===================
num_folds = 2
folds = StratifiedKFold(n_splits=num_folds)
# division entrenamiento y prueba
for train_index, test_index in folds.split(one_dimensional, y):
    x_train, x_test = one_dimensional[train_index], one_dimensional[test_index]
    y_train, y_test = y[train_index], y[test_index]
# =================== Población inicial (padres) ===================
slice_weights = [0,1,2,3]
slice_delay   = [4,5,6,7]
weights = obj_ev_dif.inicializacion(LI=-999.99, LS=999.99)  #pesos iniciales (padres)
delays = obj_ev_dif.inicializacion(LI=0.01, LS=19.99)       #delays iniciales (padres)
padres = np.column_stack([weights,delays]) #combinando pesos y delays
error_padres = np.zeros((Np,1))
error_crossover = np.zeros((Np,1))
# =================== Fase de Entrenamiento ===================
for i in range(Np):
    error = obj_ev_dif.fitness_error_srm(SRM_object=obj_srm,x_train=x_train,y_train=y_train,\
                    weight=padres[:,slice_weights][i].reshape(1,-1),delay=padres[:,slice_delay][i].reshape(1,-1))
    error_padres[i] = error

while(aux <= generaciones):
    mutado= obj_ev_dif.mutacion(padres)
    crossover= obj_ev_dif.recombinacion(padres,mutado)
    for i in range(Np):
        error = obj_ev_dif.fitness_error_srm(SRM_object=obj_srm, x_train=x_train, y_train=y_train, \
                weight=crossover[:,slice_weights][i].reshape(1, -1), delay=crossover[:,slice_delay][i].reshape(1, -1))
        error_crossover[i] = error
    padres,error_padres = obj_ev_dif.seleccion(padres,error_padres,crossover,error_crossover)

    aux += 1
# =================== Fase de prueba ===================
for j in range(Np):
    y_predicted = obj_srm.srm_train_test(x_data=x_test,y_data=y_test,\
    weight=padres[:,slice_weights][j].reshape(1,-1),delay=padres[:,slice_delay][j].reshape(1,-1))

    y_predicted = obj_srm.re_asing_y_pred(y_predicted=y_predicted)
    acc = accuracy_score(y_true=y_test, y_pred=y_predicted)
    print(y_predicted)
    print('poblacion:',j)
    print(acc)