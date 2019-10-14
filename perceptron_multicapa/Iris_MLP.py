#========================= Perceptrón Multicapa Iris Plant =========================
import pandas as pd
import numpy as np
from string import Template
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import MinMaxScaler,label_binarize
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import perceptron_multicapa.per_multicapa as MLP
from scripts import utils


#========================= Dataset
'''Cargamos el conjunto de datos y lo dividimos en entradas y salidas'''
database = 'iris plant'
iris_ds = pd.read_csv("../datasets/iris_plant/iris_plant.csv")
iris_ds["variety"] = iris_ds["variety"].replace(["Setosa"],0)
iris_ds["variety"] = iris_ds["variety"].replace(["Versicolor"],1)
iris_ds["variety"] = iris_ds["variety"].replace(["Virginica"],2)
data = iris_ds.columns.tolist()[:-1]
target = iris_ds.columns.tolist()[-1]
X = np.array(iris_ds[data])
y = np.array(iris_ds[target])
y = y.reshape(150,1)
y_bin = label_binarize((y),classes=[0,1,2]) #shape (150,3)
#=========================División entrenamiento y prueba=========================
label_names=["Setosa","Versicolor",'Virginica']
test_size = 0.3
train_size = 1 - test_size
epoch = 10000
lr = 0.05
multi_layer = MLP.MLP(X.shape[1], 3)
X_train, X_test, y_bin_train, y_bin_test = train_test_split(X,y_bin,test_size=test_size)
num_pat_train = X_train.shape[0]
num_pat_test = X_test.shape[0]

#=========================Entrenamiento y prueba=========================
y_bin_result_train = multi_layer.train_model(X_train=X_train,y_train=y_bin_train,epoch=epoch,lr=lr)
y_result_train = utils.decode_onehot(y_bin_result_train)
y_true_train = utils.decode_onehot(y_bin_train)
acc_train = (accuracy_score(y_true=y_true_train,y_pred=y_result_train)*100)
#matriz de confusion 1
plt.figure()
utils.plot_conf_matrix(y_true=y_true_train,y_pred=y_result_train,label_names=label_names)
plt.title("Matriz de confusión (entrenamiento)")
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.show()

y_bin_result_test = multi_layer.test_model(X_test=X_test)
y_predict_test = utils.decode_onehot(y_bin_result_test)
y_true_test = utils.decode_onehot(y_bin_test)
acc_test = (accuracy_score(y_true=y_true_test,y_pred=y_predict_test)*100)
#matriz de confusion 2
plt.figure()
utils.plot_conf_matrix(y_true=y_true_test,y_pred=y_predict_test,label_names=label_names)
plt.title("Matriz de confusión (prueba)")
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.show()
#=========================Resultados Entrenamiento y prueba=========================
info = Template(" =========================Información=========================\n"
                     "Nombre: $database\n"
                     "Entrenamiento: $train_size ($num_pat_train de entrenamiento)\n"
                     "Prueba: $test_size ($num_pat_test de prueba)\n"
                     "Generaciones : $generaciones\n"
                     "Tasa de aprendizaje: $lr")
info = info.substitute(
    database=database,
    train_size=train_size,
    num_pat_train=num_pat_train,
    test_size=test_size,
    num_pat_test=num_pat_test,
    generaciones=epoch,
    lr=lr)
print(info)
print('=========================Resultados entrenamiento=========================\n')
print('Entrenamiento:',num_pat_train,' patrones')
print(y_true_train)
print('Resultados entrenamiento:\n')
print(y_result_train)
print('Porcentaje de exactitud del modelo:',acc_train)
print('=========================Resultados prueba=========================\n')
print('Prueba:',num_pat_test,' patrones')
print(y_true_test)
print('Resultados prueba:')
print(y_predict_test)
print('Porcentaje de exactitud del modelo:',acc_test,'\n')


#===================aqui va mi codigal horrible
y_bin_result_test = multi_layer.train_model(X_train=X_test,y_train=y_bin_test,epoch=epoch,lr=lr)
y_result_test = utils.decode_onehot(y_bin_result_test)
y_true_test = utils.decode_onehot(y_bin_test)
acc_test = (accuracy_score(y_true=y_true_test,y_pred=y_result_test)*100)

plt.figure()
utils.plot_conf_matrix(y_true=y_true_test,y_pred=y_result_test,label_names=label_names)
plt.title("Matriz de confusión invertidos (entrenamiento)")
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.show()


y_bin_result_train = multi_layer.test_model(X_test=X_train)
y_predict_train = utils.decode_onehot(y_bin_result_train)
y_true_train = utils.decode_onehot(y_bin_train)
acc_train = (accuracy_score(y_true=y_true_train,y_pred=y_predict_train)*100)
plt.figure()
utils.plot_conf_matrix(y_true=y_true_train,y_pred=y_predict_train,label_names=label_names)
plt.title("Matriz de confusión invertidos (prueba)")
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.show()
print('=========================Resultados invirtiendo entrenamiento y prueba=========================\n')
print('Entrenamiento:',num_pat_test,' patrones')
print('exactitud del modelo:',acc_test,'\n')
print('Prueba:',num_pat_train,' patrones')
print('exactitud del modelo test',acc_train)