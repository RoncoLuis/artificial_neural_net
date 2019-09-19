"""
Neural net. Hebb Rule
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def step_function(x):
    return 1 if x >= 0 else -1

def perceptron_output(weights,bias,x):
    calculation = np.dot(weights,x)+bias
    return step_function(calculation)

#data = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
#target = np.array([])
#tabla de verdad compuerta and
tt_and = {
    "x1":[-1,-1,1,1],
    "x2":[-1,1,-1,1],
    "target":[-1,-1,-1,1]
}

tt_and = pd.DataFrame(tt_and)
data = tt_and.columns.tolist()[:-1]
target = tt_and.columns.tolist()[-1]
X = np.array(tt_and[data])
y = np.array(tt_and[target])

#funcion de propagacion
print(X.T,y)






