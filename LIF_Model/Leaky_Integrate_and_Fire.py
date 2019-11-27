"""
Luis Ronquillo
ejemplo tomado de: https://files.meetup.com/469457/spiking-neurons.pdf
"""
import numpy as np


class Leaky_Integrate_Fire:
    def __init__(self, v_rest, v_peak, simulation_time, dt, a, b, reset, theta):
        self.T = simulation_time
        self.v_rest = v_rest
        self.v_peak = v_peak
        self.dt = dt
        self.a = a
        self.b = b
        self.c = reset
        self.gain_factor = theta  # gain_factor

    def compute_I(self, dataset, weigths):
        I = np.dot(weigths, dataset.T) * self.gain_factor
        return I

    def dv_dt(self, I, V):
        v_prima = I + self.a - (self.b * V)
        return v_prima

    def LIF_train(self, x_train, y_train, weights):
        clases = np.unique(y_train)
        FR = []
        ts = np.arange(0, self.T, self.dt)
        for cl in clases:
            x = x_train[y_train == cl]
            rate = []  # Firing rate por clase
            for xi in x:
                vi = []
                fr = 0
                I = np.sum((xi * weights)) * self.gain_factor
                vt = self.v_rest
                for t in ts:
                    vi.append(vt)
                    vt = vt + self.dt * (I + self.a - (self.b * vt))
                    if vt >= self.v_peak:
                        fr += 1
                        vt = self.c
                        # print('pulso:', fr)
                rate.append(fr)
            FR.append(rate)
        return FR

    def compute_AFR_SDFR(self, FR_list):
        AFR = []
        SDFR = []
        for i in FR_list:
            AFR.append(np.mean(i))
            SDFR.append(np.std(i))
        return AFR, SDFR

    def predict_class(self,FR_list,AFR):
        clase = []
        for fr_class in FR_list:
            for fr in fr_class:
                cl = []
                for afr in AFR:
                    cl.append(abs(afr-fr))
                clase.append(np.argmin(cl))
        return clase

