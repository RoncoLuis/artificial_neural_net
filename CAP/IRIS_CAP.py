"""
Aplicando CAP
"""
import pandas as pd
import numpy as np
from scripts import utils

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

# a partir de aqui comienza el alogoritmo CAP
v_medio = np.mean(a=X,axis=0)
x_translate = X - v_medio
M = np.dot(y_bin.T,x_translate)
res = np.dot(M,x_translate.T)
res = res.T
for index,row in enumerate(res):
    max = res[index].max()
    res[index] = np.where(row < max, 0, 1)
#ingresando un nuevo patron que no estuvo en el entrenamiento
z = np.array([5.2, 3.9, 1.2, 0.1])
z_prim = z - v_medio
