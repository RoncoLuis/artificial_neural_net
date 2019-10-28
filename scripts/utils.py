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

def decode_onehot(x):
    import numpy as np
    return np.argmax(x,1)

#función 1 para mostrar matriz de confusión
def plot_conf_matrix(y_true,y_pred,label_names):
    import seaborn as sbs
    from sklearn.metrics import confusion_matrix
    confusion_ma = confusion_matrix(y_true=y_true,y_pred=y_pred)
    return sbs.heatmap(data=confusion_ma,cmap="Pastel1",annot=True,xticklabels=label_names,yticklabels=label_names)

#función 2 para mostrar matrices de confusion
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title: the text to display at the top of the matrix

    cmap:the gradient of the values displayed from matplotlib.pyplot.cm
        see http://matplotlib.org/examples/color/colormaps_reference.html
        plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm= cm,normalize= True,target_names = y_labels_vals,
                          title= best_estimator_name)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()