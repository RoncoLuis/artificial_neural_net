import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from Evolucion_Diferencial.algorithm_DE import DE
from SRM_Model.Spike_Response_Model import SRM

tiempo_inicial = time.time()
print('tiempo inicial ',time.ctime(tiempo_inicial))
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
tau = 9  # constante en el tiempo
threshold = 1
ts_inicial = 10  # tiempo inicial de simulación
ts_final = 20  # tiempo final de simulación
slice_weights = [0, 1, 2, 3]
slice_delay = [4, 5, 6, 7]
# =================== Evolución Diferencial ===================
Np = 30  # poblacion
Dim = 4  # dimensión de la poblacion (weights | delays)
Cr = 0.8  # probabilidad de cruza
F = 0.9  # factor de mutación
generaciones = 100
# =================== Declaración de Objetos ===================
obj_srm = SRM(a=a, b=b, tau=tau, threshold=threshold, ts_inicial=ts_inicial, ts_final=ts_final)
obj_ev_dif = DE(Np=Np, Dim=Dim, Cr=Cr, F=F)
# =================== Conversión dataset a señales (1-dimensional) ===================
one_dimensional = obj_srm.one_dimensional_encoding(x=X)
pesos = obj_ev_dif.inicializacion(LI=-999.99,LS=999.99)
delays = obj_ev_dif.inicializacion(LI=0.01,LS=19.99)
# =================== Division base de datos ===================
num_folds = 2
folds = StratifiedKFold(n_splits=num_folds)
# division entrenamiento y prueba
for train_index, test_index in folds.split(one_dimensional,y):
    x_train, x_test, y_train, y_test = one_dimensional[train_index], one_dimensional[test_index], y[train_index], y[test_index]

tf = obj_srm.srm_train_test(x_data=x_train,y_data=y_train,
                            weight=pesos[0].reshape(1,-1),delay=delays[0].reshape(1,-1))

targets = [12,15,18]
for cada_elemento in targets:
    for elemento in tf:
        calcula = np.argmin()