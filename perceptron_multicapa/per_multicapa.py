class MLP:
    #abstracción de los objetos MLP
    def __init__(self,input_layer,output_layer):
        #inicializar arquitectura del MLP
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.Vij = []
        self.bij = []
        self.Wjk = []
        self.bjk = []

    def train_model(self,X_train,y_train,epoch,lr=0.5):
        #funcion para inicializar la arquitectura del MLP
        import numpy as np
        import matplotlib.pyplot as plt
        from scripts import utils
        hidden_layer = (self.input_layer * 2) - 1
        # =========================Inicializando arquitectura
        self.Vij = np.random.uniform(low=-1.0,high=1.0,size=(self.input_layer,hidden_layer))
        self.bij = np.random.uniform(low=1.0,high=1.0,size=(1,hidden_layer))
        self.Wjk = np.random.uniform(low=-1.0,high=1.0,size=(hidden_layer,self.output_layer))
        self.bjk = np.random.uniform(low=1.0,high=1.0,size=(1,self.output_layer))
        mse = []
        # =========================
        for i in range(epoch):
            # ======== feedforward ========
            # == == == == capa entrada-oculta == == == ==
            hidden_layer_input_1 = np.dot(self.Vij.T,X_train.T)
            hidden_layer_input = hidden_layer_input_1.T + self.bij
            hidden_layer_sigmoid = utils.sigmoid_fun(hidden_layer_input)
            # == == == == capa oculta-salida == == == ==
            output_layer_input_1 = np.dot(hidden_layer_sigmoid,self.Wjk)
            output_layer_input = output_layer_input_1+self.bjk
            output = utils.sigmoid_fun(output_layer_input)
            # ======== backpropagation ========
            error = y_train - output
            # == == == == regreso capa salida-oculta == == == ==
            delta_output_layer = utils.sigmoid_fun(output_layer_input,der=True)
            d_output = error * delta_output_layer
            # == == == == regreso capa oculta-entrada == == == ==
            delta_hidden_layer = utils.sigmoid_fun(hidden_layer_input,der=True)
            error_hidden_layer = np.dot(d_output,self.Wjk.T)
            d_hidden_layer = error_hidden_layer * delta_hidden_layer
            # ======== update weights ========
            self.Wjk += lr * np.dot(hidden_layer_sigmoid.T,d_output)
            self.bjk += lr * np.sum(d_output,axis=0,keepdims=True)
            self.Vij += lr * np.dot(X_train.T,d_hidden_layer)
            self.bij += lr * np.sum(d_hidden_layer,axis=0,keepdims=True)
            square_error = ((1 / 2) * (np.power((error), 2)))
            mse.append(square_error.sum())
        plt.figure()
        plt.plot(mse)
        plt.title('MSE')
        plt.xlabel('Generaciones')
        plt.ylabel('Error')
        plt.show()
        return output

    def test_model(self,X_test):
        import numpy as np
        from scripts import utils
        # ======== feedforward ========
        # == == == == capa entrada-oculta == == == ==
        hidden_layer_input_1 = np.dot(self.Vij.T, X_test.T)
        hidden_layer_input = hidden_layer_input_1.T + self.bij
        hidden_layer_sigmoid = utils.sigmoid_fun(hidden_layer_input)
        # == == == == capa oculta-salida == == == ==
        output_layer_input_1 = np.dot(hidden_layer_sigmoid, self.Wjk)
        output_layer_input = output_layer_input_1 + self.bjk
        output = utils.sigmoid_fun(output_layer_input)
        return output



