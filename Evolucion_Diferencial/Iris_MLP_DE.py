#========================= MLP con Evolución Diferencial en Iris Plant =========================
import pandas as pd
import numpy as np
from Evolucion_Diferencial.MLP_DE import MLP_DE
from Evolucion_Diferencial.algorithm_DE import DE
from scripts.utils import one_hot_encode, decode_onehot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#========================= Dataset iris plant =========================
scaler = StandardScaler()
'''Cargamos el conjunto de datos y lo dividimos en entradas y salidas'''
database = 'iris plant'
iris_ds = pd.read_csv("../datasets/iris_plant/iris_plant.csv")
iris_ds["variety"] = iris_ds["variety"].replace(["Setosa"], 0)
iris_ds["variety"] = iris_ds["variety"].replace(["Versicolor"], 1)
iris_ds["variety"] = iris_ds["variety"].replace(["Virginica"], 2)
data = iris_ds.columns.tolist()[:-1]
target = iris_ds.columns.tolist()[-1]
X = np.array(iris_ds[data])
X_std = scaler.fit(X).transform(X) #dataset estandar
y = np.array(iris_ds[target])
y_bin = one_hot_encode(X_data=X, y_data=y, num_cols=3)
#========================= Fin pre-procesamiento iris plant

#========================= Constantes Perceptron Multicapa (Evolucion Diferencial) =========================
Np = 30 #population
Cr = 0.8 #crossover ratio
F = 0.9 #mutacion
Li = -100.0 #limite inferior
Ls = 100.0 #limite superior
generaciones = 1000 #Generaciones evolucion diferencial
folders = 2
aux = 0
#========================= Fin constantes

#========================= Inicializando Objetos (clases) =========================
perceptron = MLP_DE(input_neurons=X.shape[1], output_neurons=y_bin.shape[1]) #arquitectura 4 neuronas entrada y 3 salida
indices = perceptron.calcula_indices() #inidices para calcular la dimensionalidad
evolucion_diferencial = DE(Np=Np, Dim=indices["dim"], Cr=Cr, F=F) #inicializando evolucion diferencial
#========================= Fin inicializacion

#========================= KFolds =========================
kfold = StratifiedKFold(n_splits=folders)

for train_index, test_index in kfold.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_bin[train_index], y_bin[test_index]
#========================= Fin KFolds

#========================= Algoritmo Evolución Diferencial =========================
padres = evolucion_diferencial.inicializacion(LI=Li, LS=Ls) #generando la poblacion inicial
error_en_padres = np.zeros((Np,1))
error_en_crossover = np.zeros((Np,1))

for i in range(Np):
    error_padres = evolucion_diferencial.fitness_error_mlp(MLP_object=perceptron,X=X_train, y=y_train, weights=padres[i].reshape(1,-1))
    error_en_padres[i] = error_padres

while(aux <= generaciones):
    #print('==== generacion:',aux,' =====')
    mutado = evolucion_diferencial.mutacion(padres)
    crossover = evolucion_diferencial.recombinacion(padres,mutado)

    for i in range(crossover.shape[0]):
        error_crossover = evolucion_diferencial.fitness_error_mlp(MLP_object=perceptron,X=X_train, y=y_train, weights=crossover[i].reshape(1,-1))
        error_en_crossover[i] = error_crossover

    padres,error_en_padres = evolucion_diferencial.seleccion(padres,error_en_padres,crossover,error_en_crossover)

    aux += 1
#========================= Fin Algoritmo Evolución Diferencial

#========================= data de prueba con la ultima generacion DE =========================
for i in range(Np):
    prediction = perceptron.train_model(X_train=X_train, y_train=y_train, DE_population=padres[i].reshape(1, -1))
    print(accuracy_score(decode_onehot(y_train), decode_onehot(prediction)))