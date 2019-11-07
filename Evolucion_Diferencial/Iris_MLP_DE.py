#========================= MLP con Evolución Diferencial en Iris Plant =========================
import pandas as pd
import numpy as np
from Evolucion_Diferencial.MLP_DE import MLP_DE
from Evolucion_Diferencial.algorithm_DE import DE
from scripts.utils import one_hot_encode, decode_onehot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#========================= Dataset iris plant
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
y = np.array(iris_ds[target])
y_bin = one_hot_encode(X_data=X, y_data=y, num_cols=3)
X_std = scaler.fit(X).transform(X)
#========================= Fin pre-procesamiento iris plant

#========================= Constantes Perceptron con Evolución diferencial
Np = 30 #population
Cr = 0.7 #crossover ratio
F = 0.8 #mutacion
Li = -10.0 #limite inferior
Ls = 10.0 #limite superior
generaciones = 1000 #Generaciones evolucion diferencial

perceptron = MLP_DE(input_neurons=X.shape[1], output_neurons=y_bin.shape[1]) #inicializando objeto perceptron
indices = perceptron.calcula_indices()
evolucion_diferencial = DE(Np=Np, Dim=indices["dim"], Cr=Cr, F=F) #inicializando evolucion diferencial

padres = evolucion_diferencial.inicializacion(LI=Li,LS=Ls)

#obteniendo 1er error de los primeros padres antes de entrar al algoritmo de evolucion diferencial
error_en_padres = np.zeros((Np,1))
for i in range(Np):
    error_padres = evolucion_diferencial.fitness_error_mlp(MLP_object=perceptron,X=X, y=y_bin, weights=padres[i].reshape(1,-1))
    error_en_padres[i] = error_padres

error_en_crossover = np.zeros((Np,1))

aux = 0
while(aux <= generaciones):
    mutado = evolucion_diferencial.mutacion(padres)
    crossover = evolucion_diferencial.recombinacion(padres,mutado)

    for i in range(crossover.shape[0]):
        error_crossover = evolucion_diferencial.fitness_error_mlp(MLP_object=perceptron,X=X, y=y_bin, weights=crossover[i].reshape(1,-1))
        error_en_crossover[i] = error_crossover

    padres,error_en_padres = evolucion_diferencial.seleccion(padres,error_en_padres,crossover,error_en_crossover)
    #for j in range(Np):
    aux += 1

prediction = perceptron.train_model(X_train=X, y_train=y_bin, DE_population=padres[error_en_padres.argmin()].reshape(1, -1))
print(accuracy_score(y, decode_onehot(prediction)))

