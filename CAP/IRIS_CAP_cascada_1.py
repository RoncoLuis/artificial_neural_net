"""
Luis Ronquillo
Clasificador Asociativo de Patrones (CAP)
base de datos : iris plant
#TODO pag.91 hands-on machine learning ->Plot Roc Curve
#TODO pag.86 hands-on machine learning -> precision and recall scores
"""
import pandas as pd
import numpy as np
from scripts import utils
from CAP.algorithm_CAP import CAP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  confusion_matrix,precision_score,recall_score,roc_curve,mean_squared_error

iris_ds = pd.read_csv("../datasets/iris_plant/iris_plant.csv")
# ============= pre-procesamiento ============= #
# ============= Setosa => 0 | virginica,versicolor => 1
iris_ds["variety"] = iris_ds["variety"].replace(["Setosa"],0)
iris_ds["variety"] = iris_ds["variety"].replace(["Versicolor"],1)
iris_ds["variety"] = iris_ds["variety"].replace(["Virginica"],1)
data = iris_ds.columns.tolist()[:-1]
target = iris_ds.columns.tolist()[-1]
X = np.array(iris_ds[data])
y = np.array(iris_ds[target])
# ============= Setosa => [1,0,0] | virginica,versicolor => [0,1,0]
y_bin = utils.one_hot_encode(X_data=X,y_data=y,num_cols=3)
table_score_train = []
table_score_test = []
table_mse_train = []
table_mse_test = []
for i in range(35):
# ============= K-Folds ============= #
    kfold = StratifiedKFold(n_splits=2,shuffle=True)
    #Este for se va ejecutar n_splits veces adentro debe ir el CAP
    score_train = []
    score_test = []
    mse_train = []
    mse_test = []
    for train_index,test_index in kfold.split(X,y):
        X_train,X_test,y_train,y_test = X[train_index],X[test_index],\
                                        y_bin[train_index],y_bin[test_index]
        #print("train",X_train.shape)
        #print("test",X_test.shape)
        cap = CAP(X_train=X_train,y_train_bin=y_train)
        y_train_pred = cap.recall()
        correct_train = sum(y_train_pred == y_train)
        score_train.append(correct_train/len(y_train_pred))
        mse_train.append(mean_squared_error(y_train,y_train_pred))
        #print(confusion_matrix(utils.decode_onehot(y_train),utils.decode_onehot(y_train_pred)))
        #print("Precision or accuracy",precision_score(y_true=utils.decode_onehot(y_train),y_pred=utils.decode_onehot(y_train_pred)))
        #print("Recall or sensitivity",recall_score(y_true=utils.decode_onehot(y_train), y_pred=utils.decode_onehot(y_train_pred)))
        y_pred = cap.recall_new(z=X_test)
        n_correct = sum(y_pred == y_test)
        score_test.append(n_correct/len(y_pred))
        mse_test.append(mean_squared_error(y_test,y_pred))
    table_score_train.append(score_train)
    table_score_test.append(score_test)
    table_mse_train.append(mse_train)
    table_mse_test.append(mse_test)
# ============= Termina parte 1 en cascada =============
#Ahora obtenemos los resultados obtenidos por el primer CAP y continuamos con la 2da parte del cascada
#cargando la base de datos iris plant original, donde se reemplazara con los indices que nos dejo el CAP
cap_iris = pd.read_csv("../datasets/iris_plant/iris_plant.csv")
filtro_train = cap_iris.iloc[train_index] #indice de entrenamiento obtenido del CAP anterior
filtro_test = cap_iris.iloc[test_index] #indice de prueba obtenido del CAP anterior
#retirando los campos que el  CAP anterior identifico como setosa
filtro_train = filtro_train[filtro_train.variety != "Setosa"]
filtro_test = filtro_test[filtro_test.variety != "Setosa"]
#concatenando los resultados para formar un nuevo DS
cap_iris = pd.concat([filtro_train,filtro_test],axis=0)
# ============= Reemplazo con clase nube y clase secundaria =============
cap_iris["variety"] = cap_iris["variety"].replace(["Versicolor"],0)
cap_iris["variety"] = cap_iris["variety"].replace(["Virginica"],1)
cap_data = cap_iris.columns.tolist()[:-1]
cap_target = cap_iris.columns.tolist()[-1]
cap_X = np.array(cap_iris[cap_data])
cap_y = np.array(cap_iris[cap_target])
# ============= Metodo en cascada 2da parte =============
cap_y_bin = utils.one_hot_encode(X_data=cap_X,y_data=cap_y,num_cols=3)
cap_table_score_train = []
cap_table_score_test = []
cap_table_mse_train = []
cap_table_mse_test = []
for i in range(35):
# ============= K-Folds ============= #
    cap_kfold = StratifiedKFold(n_splits=2,shuffle=True)
    #Este for se va ejecutar n_splits veces adentro debe ir el CAP
    cap_score_train = []
    cap_score_test = []
    cap_mse_train = []
    cap_mse_test = []
    for cap_train_index,cap_test_index in cap_kfold.split(cap_X,cap_y):
        cap_X_train,cap_X_test,cap_y_train,cap_y_test = cap_X[cap_train_index],cap_X[cap_test_index],\
                                        cap_y_bin[cap_train_index],cap_y_bin[cap_test_index]
        #print("train",X_train.shape)
        #print("test",X_test.shape)
        cap_2 = CAP(X_train=cap_X_train,y_train_bin=cap_y_train)
        cap_y_train_pred = cap_2.recall()
        cap_correct_train = sum(cap_y_train_pred == cap_y_train)
        cap_score_train.append(cap_correct_train/len(cap_y_train_pred))
        cap_mse_train.append(mean_squared_error(cap_y_train,cap_y_train_pred))
        cap_y_pred = cap_2.recall_new(z=cap_X_test)
        cap_n_correct = sum(cap_y_pred == cap_y_test)
        cap_score_test.append(cap_n_correct/len(cap_y_pred))
        cap_mse_test.append(mean_squared_error(cap_y_test,cap_y_pred))
    cap_table_score_train.append(cap_score_train)
    cap_table_score_test.append(cap_score_test)
    cap_table_mse_train.append(cap_mse_train)
    cap_table_mse_test.append(cap_mse_test)

# ============= Exportando los resultados  ============= #
#Exportando las tablas de resultados para el scrip de estadisticos
def export_to_csv(X,filename,path="../datasets/iris_plant/resultados/"):
    extension = ".csv"
    X = pd.DataFrame(X)
    X.to_csv(str(path+filename+extension),index=False)

export_to_csv(table_score_test,filename="table_score_test")
export_to_csv(table_score_train,filename="table_score_train")
export_to_csv(table_mse_train,filename="table_mse_train")
export_to_csv(table_mse_test,filename="table_mse_test")
export_to_csv(cap_table_score_test,filename="cap_table_score_test")
export_to_csv(cap_table_score_train,filename="cap_table_score_train")
export_to_csv(cap_table_mse_train,filename="cap_table_mse_train")
export_to_csv(cap_table_mse_test,filename="cap_table_mse_test")