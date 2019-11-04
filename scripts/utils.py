def one_hot_encode(X_data,y_data,num_cols):
    #función para codificar los targets en binarios
    #num_cols será el número de columnas como resultado de binarizar
    import numpy as np
    one_hot_labels = np.zeros((X_data.shape[0], num_cols))
    for i in range(X_data.shape[0]):
        one_hot_labels[i, y_data[i]] = 1
    return one_hot_labels

def sigmoid_fun(x,der=False):
    import numpy as np
    if der == False:
        result = 1/(1+np.exp(-x))
    else:
        result = 1/(1+np.exp(-x)) * ( 1 - (1/(1+np.exp(-x))) )
    return result

def sigmoid(t):
    import math as m
    return 1/(1+m.exp(-t))

def decode_onehot(x):
    import numpy as np
    return np.argmax(x,1)

#función 1 para mostrar matriz de confusión
def plot_conf_matrix(y_true,y_pred,label_names):
    import seaborn as sbs
    from sklearn.metrics import confusion_matrix
    confusion_ma = confusion_matrix(y_true=y_true,y_pred=y_pred)
    return sbs.heatmap(data=confusion_ma,cmap="Pastel1",annot=True,xticklabels=label_names,yticklabels=label_names)
