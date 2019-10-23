import pandas as pd
import numpy as np
from CAP.algorithm_CAP import CAP
X = np.array([[-0.33,0.69,0,1,1,0,0.8,0,0.8],[1,0.64,0,0,1,0,0.8,-1,0.25],\
              [-0.33,0.94,1,0,1,0,0.8,1,0.31]])
y = np.array([[0,1],[0,1],[1,0]])
cap = CAP(X_train=X,y_train_bin=y)
y_result = cap.recall()

z = np.array([[1,0.78,1,1,1,0,0.6,0,0.13],[-0.33,0.67,1,1,0,0,0.8,-1,0.5],\
              [-0.33,0.5,1,1,0,-1,0.8,0,0.88]])
y_predict = cap.recall_new(z)