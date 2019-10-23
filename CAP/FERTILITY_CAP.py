"""
Luis Ronquillo
Clasificador Asociativo de Patrones (CAP)
base de datos : Fertility
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from scripts import utils
from CAP.algorithm_CAP import CAP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.preprocessing import StandardScaler

#base de datos original
original_fertility = pd.read_csv("../datasets/fertility/fertility.csv")
fertility_ds = original_fertility.copy()
fertility_ds["diagnosis"] = fertility_ds["diagnosis"].replace(["N"],0) #normal [1,0]
fertility_ds["diagnosis"] = fertility_ds["diagnosis"].replace(["O"],1) #altered [0,1]
normal = fertility_ds[fertility_ds["diagnosis"] == 0]
altered = fertility_ds[fertility_ds["diagnosis"] == 1]
scaler = StandardScaler()
data = fertility_ds.columns.tolist()[:-1]
target = fertility_ds.columns.tolist()[-1]
X = np.array(fertility_ds[data]) #sin normalizar
X_std = scaler.fit(X).transform(X) #normalizado
y = np.array(fertility_ds[target]).reshape(X.shape[0],1)
y_bin = utils.one_hot_encode(X_data=X,y_data=y,num_cols=2)

table_score_train = []
table_score_test = []
table_mse_train = []
table_mse_test = []

for i in range(35):
    kfold = StratifiedKFold(n_splits=2,shuffle=True)
    score_train = []
    score_test = []
    mse_train = []
    mse_test = []
    for train_index,test_index in kfold.split(X,y):
        X_train,X_test,y_train,y_test = X_std[train_index],X_std[test_index],\
                                        y_bin[train_index],y_bin[test_index]
        cap = CAP(X_train=X_train,y_train_bin=y_train)
        y_train_predicted = cap.recall()
        score_train.append(accuracy_score(y_true=utils.decode_onehot(y_train),\
                                          y_pred=utils.decode_onehot(y_train_predicted)))
        mse_train.append(mean_squared_error(y_train,y_train_predicted))


        y_test_predicted = cap.recall_new(z=X_test)
        score_test.append(accuracy_score(y_true=utils.decode_onehot(y_test), \
                                          y_pred=utils.decode_onehot(y_test_predicted)))
        mse_test.append(mean_squared_error(y_test, y_test_predicted))
        #Cuando finaliza una iteración se guarda los scores en tablas generales
    table_score_train.append(score_train)
    table_score_test.append(score_test)
    table_mse_train.append(mse_train)
    table_mse_test.append(mse_test)


# Exportando las tablas de resultados para el scrip de estadisticos
def export_to_csv(X, filename, path="../datasets/fertility/resultados/"):
        extension = ".csv"
        X = pd.DataFrame(X)
        X.to_csv(str(path + filename + extension), index=False)


export_to_csv(table_score_test, filename="table_score_test")
export_to_csv(table_score_train, filename="table_score_train")
export_to_csv(table_mse_train, filename="table_mse_train")
export_to_csv(table_mse_test, filename="table_mse_test")


X_train = pd.DataFrame(data=X_train,columns=data)
y_train_predicted = pd.DataFrame(data=utils.decode_onehot(y_train_predicted))
X_train["diagnosis"] = y_train_predicted
normal_train = X_train[X_train["diagnosis"] == 0]
altered_train = X_train[X_train["diagnosis"] == 1]

X_test = pd.DataFrame(data=X_test,columns=data)
y_test_predicted = pd.DataFrame(data=utils.decode_onehot(y_test_predicted))
X_test["diagnosis"] = y_test_predicted
normal_test = X_test[X_test["diagnosis"] == 0]
altered_test = X_test[X_test["diagnosis"] == 1]