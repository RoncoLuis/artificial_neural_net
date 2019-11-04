"""
Luis Ronquillo
Perceptrón multicapa usando DE
"""
import numpy as np
from scripts.utils import sigmoid_fun
from sklearn.metrics import accuracy_score,mean_squared_error

class MLP_DE :
    def __init__(self, input_neurons, output_neurons):
        self.input_layer = input_neurons
        self.output_layer = output_neurons
        self.Vij = []
        self.bij = []
        self.Wjk = []
        self.bjk = []

    def train_model(self, X_train, y_train, DE_population):
        indices_DE = self.calcula_indices()
        #Ajustando vector de entrada añadiendio columna bias a X_train
        self.Vij = np.array(DE_population[0, 0:indices_DE["vij"]])
        self.bij = np.array(DE_population[0, indices_DE["vij"]:indices_DE["bij"]])
        self.Wjk = np.array(DE_population[0, indices_DE["bij"]:indices_DE["wjk"]])
        self.bjk = np.array(DE_population[0, indices_DE["wjk"]:indices_DE["bjk"]])
        #reshape de los pesos
        self.Vij = self.Vij.reshape(indices_DE["input_layer"],indices_DE["hidden_layer"])
        self.bij = self.bij.reshape(1, indices_DE["hidden_layer"])
        self.Wjk = self.Wjk.reshape(indices_DE["hidden_layer"],indices_DE["output_layer"])
        self.bjk = self.bjk.reshape(1, indices_DE["output_layer"])
        # ============ feedforward ============
        # ============ capa entrada-oculta ============
        input_to_hidden = np.dot(self.Vij.T, X_train.T)
        input_to_hidden_bias = input_to_hidden.T + self.bij
        input_to_hidden_sigmoid = sigmoid_fun(input_to_hidden_bias)
        # ============ capa oculta-salida ============
        hidden_to_output = np.dot(input_to_hidden_sigmoid, self.Wjk)
        hidden_to_output_bias = hidden_to_output+self.bjk
        output = sigmoid_fun(hidden_to_output_bias)
        return output

    def calcula_indices(self):
        input_neurons = self.input_layer
        hidden_layer = ((self.input_layer*2)-1)
        output_neurons = self.output_layer
        vij = input_neurons*hidden_layer
        bij = 1 * hidden_layer
        bjk = 1 * output_neurons
        wjk = hidden_layer*output_neurons
        index_dict = {
            "input_layer": input_neurons,
            "hidden_layer": hidden_layer,
            "output_layer": output_neurons,
            "vij": vij,
            "bij": vij+bij,
            "wjk": vij+bij+wjk,
            "bjk": vij+bij+wjk+bjk,
            "dim": vij+bij+wjk+bjk
        }
        return index_dict

