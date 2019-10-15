"""
Aplicando CAP
"""
import pandas as pd
import numpy as np
from scripts import utils
from CAP.algorithm_CAP import CAP

iris_ds = pd.read_csv("../datasets/iris_plant/iris_plant.csv")
#metodología en cascada
iris_ds["variety"] = iris_ds["variety"].replace(["Setosa"],0)
iris_ds["variety"] = iris_ds["variety"].replace(["Versicolor"],1)
iris_ds["variety"] = iris_ds["variety"].replace(["Virginica"],1)
data = iris_ds.columns.tolist()[:-1]
target = iris_ds.columns.tolist()[-1]
X = np.array(iris_ds[data])
y = np.array(iris_ds[target])
y_bin = utils.one_hot_encode(X_data=X,y_data=y,num_cols=2)

#ingresando un nuevo patron que no estuvo en el entrenamiento
z = np.array([5.2, 3.9, 1.2, 0.1])

cap = CAP(X_train=X,y_train_bin=y_bin)
y_result = cap.recall()
z_result = cap.recall_new(z)