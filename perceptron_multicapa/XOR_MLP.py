"""
Luis Ronquillo
date : 09/19/2019
Perceptron multicapa "Multi Layer Perceptron" (MLP)
"""
# importing the library
import numpy as np

# creating the input array
#X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
X=np.array([[0,0],[0,1],[1,0],[1,1]])
print ('\n Entrada:')
print(X)

# creating the output array
y=np.array([[0],[1],[1],[0]])
print ('\n Salida esperada:')
print(y)

# defining the Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Cambiar el valor de salida, lo reemplaza por 0 y 1, Según el umbral
def replace_(umbral,y_predict):
    y_predict[np.where(y_predict >umbral )] = 1
    y_predict[np.where(y_predict <= umbral)] = 0
    return y_predict

# initializing the variables
epoch=10000 # number of training iterations
lr=0.5 # learning rate
inputlayer_neurons = X.shape[1] # number of features in data set
hiddenlayer_neurons = 3 # number of hidden layers neurons
output_neurons = 1 # number of neurons at output layer

# initializing weight and bias
wh=np.random.uniform(low=-1.0,high=1.0,size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(low=-1.0,high=1.0,size=(1,hiddenlayer_neurons))
wout=np.random.uniform(low=-1.0,high=1.0,size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(low=-1.0,high=1.0,size=(1,output_neurons))

# training the model
for i in range(epoch):
    #========================= Feedforward =========================
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh

    hiddenlayer_activations = sigmoid(hidden_layer_input)

    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)

    #========================= Backpropagation =========================
    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

    #========================= Update weights =========================
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print ('\n Salida del modelo:')
print ("Salida real",output)

def test_model(X_test):
    #Esta función aplica los pasos del feedforward para el test del modelo
    zin_j_pt1 = np.dot(X_test,wh)
    zin_j = zin_j_pt1+bh

    zj = sigmoid(zin_j_pt1)

    yin_k_pt1 = np.dot(zj,wout)
    y_in_k = yin_k_pt1+bout

    yk = sigmoid(y_in_k)
    return yk
#========================= Probando XOR =========================
X_test=np.array([[0,1],[1,1]])
y_predict = test_model(X_test)
print("Salida test (reales)",y_predict)
y_predict = replace_(0.5,y_predict)
print("Salida con reemplazo",y_predict)



