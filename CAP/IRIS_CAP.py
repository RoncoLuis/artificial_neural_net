"""
Luis Ronquillo
Clasificador Asociativo de Patrones (CAP)
base de datos : iris plant
#TODO pag.91 hands-on machine learning ->Plot Roc Curve
#TODO pag.86 hands-on machine learning -> precision and recall scores
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# ============= Setosa => [1,0] | virginica,versicolor => [0,1]
y_bin = utils.one_hot_encode(X_data=X,y_data=y,num_cols=2)
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
        print("train",X_train.shape)
        print("test",X_test.shape)
        cap = CAP(X_train=X_train,y_train_bin=y_train)
        y_train_pred = cap.recall()
        correc_train = sum(y_train_pred == y_train)
        score_train.append(correc_train/len(y_train_pred))
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
