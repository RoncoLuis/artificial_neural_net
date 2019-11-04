#========================= MLP con Evolución Diferencial en Iris Plant =========================
import pandas as pd
import numpy as np
from Evolucion_Diferencial.MLP_DE import MLP_DE
from Evolucion_Diferencial.algorithm_DE import DE
from scripts.utils import one_hot_encode, decode_onehot
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score,mean_squared_error

#========================= Dataset iris plant
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
#========================= Fin pre-procesamiento iris plant

#========================= Constantes Perceptron con Evolución diferencial
Np = 20 #population
Cr = 0.7 #crossover ratio
F = 0.15 #mutacion
Li = -1.14 #limite inferior
Ls = 1.21 #limite superior
De_generations = 100 #Generaciones evolucion diferencial
mlp_DE = MLP_DE(input_neurons=X.shape[1], output_neurons=y_bin.shape[1])
indexes = mlp_DE.calcula_indices()
De_ = DE(Np=Np, Dim=indexes["dim"], Cr=Cr, F=F)
best_population = []
De_init_population = De_.generate_initial_pob(LI=Li, LS=Ls)
weights = np.array(De_.DE_function(pob_init=De_init_population, generations=De_generations))

#obteniendo los mejores resultados
for j in range(35):
    print('===== experimento[', j, '] ===== ')
    # separando la base de datos en numero de folds
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    for train_index, test_index in kfold.split(X, y):
        #print("train index:", train_index, " test index:", test_index)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], \
                                           y_bin[train_index], y_bin[test_index]

    for i in range(Np):
        result_train = mlp_DE.train_model(X_train=X_train, y_train=y_train, DE_population=weights[i].reshape(1, -1))
        acc = accuracy_score(y_true=decode_onehot(y_train),y_pred=decode_onehot(result_train))
        error = mean_squared_error(y_true=decode_onehot(y_train),y_pred=decode_onehot(result_train))
        if(acc > 0.6):
            best_population.append(weights[i].reshape(1, -1))
            print("iteracion",j,' en weights[', i, ']')
            print("accuracy:", acc, " error:", error)
            result_test = mlp_DE.train_model(X_train=X_test, y_train=y_test, DE_population=weights[i].reshape(1, -1))
            acc_test = accuracy_score(y_true=decode_onehot(y_test), y_pred=decode_onehot(result_test))
            error_test = mean_squared_error(y_true=decode_onehot(y_test), y_pred=decode_onehot(result_test))
            if(acc_test > 0.6):
                print("accuracy_test:", acc_test, " error_test:", error_test)



