"""
@author : Luis Ronquillo
@date   : 27/11/2019
============= Spike Response Model =============
"""
import numpy as np
class SRM:
    def __init__(self,a,b,tau,threshold,ts_inicial,ts_final):
        """
        :param a:
        :param b:
        :param tau:
        :param threshold:
        :param ts_inicial:
        :param ts_final:
        """
        self.a = a
        self.b = b
        self.tau = tau
        self.threshold = threshold
        self.ts_inicial = ts_inicial
        self.ts_final = ts_final

    def temp_coding(self, x, M, m):
        """
        temporal enconding: convierte los elementos de X a señales de entrada
        :param x: elemento original del dataset
        :param M: elemento mayor por caracteristica
        :param m: elemento menor por caracteristica
        :return:  elemento convertido a señal
        """
        r = M-m
        yf = (self.b-self.a)/r * x + (self.a*M)-(self.b*m)/r
        return yf

    def one_dimensional_encoding(self,x):
        """
        :param x: dataset original
        :return:  dataset convertido a señal
        """
        M,m = self.ranges_per_column(x)
        fila, columna = x.shape[0], x.shape[1]
        conversion = np.zeros(shape=(fila, columna))
        for i in range(fila):
            for j in range(columna):
                conversion[i, j] = self.temp_coding(x=x[i, j], M=M[j], m=m[j])
        return conversion

    def ranges_per_column(self,x):
        """
        Obtener los rangos por columnas
        :param x: dataset original
        :return:  valores Maximos y minimos por columna (M,m)
        """
        M = np.max(x,axis=0)
        m = np.min(x,axis=0)
        return M,m

    def srm_train_test(self,x_data,y_data,weight,delay):
        """
        :param x_data: dataset de entrenamiento o prueba
        :param y_data: targets del dataset
        :param weight: vector de pesos
        :param delay:  vector de retrasos
        :return: tiempo de disparo de cada uno
        """
        clases = np.unique(y_data)
        ts = np.arange(self.ts_inicial, self.ts_final, 1)  # tiempo de simulacion
        TF = [] #tiempo de disparo
        for cl in clases:
            x = x_data[y_data == cl]
            for ti in x:
                for t in ts:
                    Yi=self.calula_yi(t=t,ti=ti,di=delay)
                    Vt = np.dot(weight,Yi.T)
                    if Vt >= self.threshold:
                        TF.append(t)
                        break
                    if t >= ts[-1]:
                        TF.append(1000)
        return np.array(TF)

    def calula_yi(self,t,ti,di):
        """
        :param t: tiempo actual de simulacion [10ms-20ms]
        :param ti: elemento transformada a señal temporal
        :param di: delay del elemento
        :return: resultado de la opera yi
        """
        return (t-ti-di)/self.tau * np.exp(1-((t-ti-di)/self.tau))

    def re_asing_y_pred(self,y_predicted):
        """
        :param y_predicted:
        :return:
        """
        y_predicted = np.where(y_predicted == 10,12, y_predicted)
        y_predicted = np.where(y_predicted == 11,12, y_predicted)
        y_predicted = np.where(y_predicted == 13,12, y_predicted)
        y_predicted = np.where(y_predicted == 14,15, y_predicted)
        y_predicted = np.where(y_predicted == 16,15, y_predicted)
        y_predicted = np.where(y_predicted == 17,18, y_predicted)
        y_predicted = np.where(y_predicted == 19,18, y_predicted)
        y_predicted = np.where(y_predicted == 20,18, y_predicted)
        return y_predicted