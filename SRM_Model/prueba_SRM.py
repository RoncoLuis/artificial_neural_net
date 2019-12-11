import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
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
generaciones = 10
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

# =================== Función Fase de Entrenamiento ===================
def train_phase(obj_srm, obj_ev_dif, x_train, y_train):
    aux = 0  # contador auxiliar generaciones
    weights = obj_ev_dif.inicializacion(LI=-999.99, LS=999.99)  # pesos iniciales (padres)
    delays = obj_ev_dif.inicializacion(LI=0.01, LS=9.0)  # delays iniciales (padres)
    padres = np.column_stack([weights, delays])  # combinando pesos y delays
    error_padres = np.zeros((Np, 1))
    error_crossover = np.zeros((Np, 1))

    for i in range(Np):
        error = obj_ev_dif.fitness_error_srm(SRM_object=obj_srm, x_train=x_train, y_train=y_train, \
                                             weight=padres[:, slice_weights][i].reshape(1, -1),
                                             delay=padres[:, slice_delay][i].reshape(1, -1))
        error_padres[i] = error

    while aux <= generaciones:
        mutado = obj_ev_dif.mutacion(padres)
        crossover = obj_ev_dif.recombinacion(padres, mutado)
        for i in range(Np):
            error = obj_ev_dif.fitness_error_srm(SRM_object=obj_srm, x_train=x_train, y_train=y_train, \
                                                 weight=crossover[:, slice_weights][i].reshape(1, -1),
                                                 delay=crossover[:, slice_delay][i].reshape(1, -1))
            error_crossover[i] = error
        padres, error_padres = obj_ev_dif.seleccion(padres, error_padres, crossover, error_crossover)

        aux += 1
    return padres, error_padres


# =================== Función Fase de prueba ===================
def test_phase(obj_srm, padres, error_padres, x_test, y_test):
    error_min = np.argmin(error_padres)

    y_predicted = obj_srm.srm_train_test(x_data=x_test, y_data=y_test, \
                                         weight =padres[:, slice_weights][error_min].reshape(1, -1),
                                         delay  =padres[:, slice_delay][error_min].reshape(1, -1))

    y_predicted = obj_srm.re_asing_y_pred(y_predicted=y_predicted)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_predicted)
    cm = confusion_matrix(y_true=y_test, y_pred=y_predicted)
    return accuracy,cm

accs   = []
m_conf = []
for i in range(35):
    # =================== Microexperimento 1 ===================
    nuevos_pesos,error = train_phase(obj_srm=obj_srm,obj_ev_dif=obj_ev_dif,x_train=x_train,y_train=y_train)
    acc_m1,cm_m1 = test_phase(obj_srm=obj_srm,padres=nuevos_pesos,error_padres=error,x_test=x_test,y_test=y_test)
    # =================== Microexperimento 2 ===================
    new_weights,error_n = train_phase(obj_srm=obj_srm,obj_ev_dif=obj_ev_dif,x_train=x_test,y_train=y_test)
    acc_m2,cm_m2 = test_phase(obj_srm=obj_srm,padres=new_weights,error_padres=error_n,x_test=x_train,y_test=y_train)
    print(i,',',acc_m1,',',acc_m2)
    accs.append([acc_m1,acc_m2])
    m_conf.append([cm_m1,cm_m2])

tiempo_final = time.time()
print('tiempo final ',time.ctime(tiempo_final))





















