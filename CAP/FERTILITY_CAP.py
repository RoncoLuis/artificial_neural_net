"""
Luis Ronquillo
Clasificador Asociativo de Patrones (CAP)
base de datos : Fertility
"""
# =================== # Imports # ===================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts import utils
from CAP.algorithm_CAP import CAP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# =================== # Carga y acondicionamiento DS # ===================
fertility_ds = pd.read_csv("../datasets/fertility/fertility.csv")
#Diagnosis N = normal = 0 , O = altered = 1
fertility_ds["diagnosis"] = fertility_ds["diagnosis"].replace(["N"],1)
fertility_ds["diagnosis"] = fertility_ds["diagnosis"].replace(["O"],0)
data = fertility_ds.columns.tolist()[:-1]
target = fertility_ds.columns.tolist()[-1]
X = np.array(StandardScaler().fit_transform(fertility_ds[data])) #normalización
y = np.array(fertility_ds[target]).reshape(X.shape[0],1)
#TODO encontrar caracteristicas mas correlacionadas
# ======= one hot encode
y_bin = utils.one_hot_encode(X_data=X,y_data=y,num_cols=2)
# =================== # Tabla de resultados # ===================
table_score_train = []
table_score_test = []
table_mse_train = []
table_mse_test = []
# =================== # CAP # ===================
for i in range(35):
    kfold = StratifiedKFold(n_splits=10,shuffle=True)
    score_train = []
    score_test = []
    mse_train = []
    mse_test = []
    for train_index,test_index in kfold.split(X,y):
        X_train,X_test,y_train,y_test = X[train_index],X[test_index],\
                                        y_bin[train_index],y_bin[test_index]
        cap = CAP(X_train=X_train,y_train_bin=y_train)
        y_train_pred = cap.recall()
        correct_train = sum(y_train_pred == y_train)
        score_train.append(correct_train/len(y_train_pred))
        mse_train.append(mean_squared_error(y_train,y_train_pred))
        y_test_pred = cap.recall_new(z=X_test)
        correct_test = sum(y_test_pred == y_test)
        score_test.append(correct_test/len(y_test_pred))
        mse_test.append(mean_squared_error(y_test,y_test_pred))
    table_score_train.append(score_train)
    table_score_test.append(score_test)
    table_mse_train.append(mse_train)
    table_mse_test.append(mse_test)
# =================== # Estadísticos # ===================
table_score_train = np.array(table_score_train)
table_score_test = np.array(table_score_test)
table_mse_train = np.array(table_mse_test)
table_mse_test = np.array(table_mse_test)


