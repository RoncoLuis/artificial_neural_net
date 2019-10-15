import numpy as np
#ejercicio de prueba CAP
x1 = np.array([[6,5,2]])
x1 = x1.reshape(3,1)
y1 = np.array([1,0])
y1 = y1.reshape(2,1)
x2 = np.array([[-4,11,8]])
x2 = x2.reshape(3,1)
y2 = np.array([0,1])
y2 = y2.reshape(2,1)
v_promedio = (x1+x2)/2
#desplazar los vectores con respecto a v_promedio
x1_prim = x1 - v_promedio
x2_prim = x2 - v_promedio
#calcular Memoria_1 y Memoria_2
M1 = np.dot(y1,x1_prim.T)
M2 = np.dot(y2,x2_prim.T)
#suma de las memoria
M = M1 + M2
#traslado de vectores
t_x1 = np.dot(M,x1_prim)
t_x2 = np.dot(M,x2_prim)

def traslado(arreglo):
    maximo = arreglo.max()
    new = []
    for i in arreglo:
        if i < maximo:
            new.append(0)
        else:
            new.append(1)
    return new
#añadiendo un vector que no estuvo en el entrenamiento
z = np.array([[4,7,-1]])
z = z.reshape(3,1)
yz = np.array([1,0])
yz = y1.reshape(2,1)
z_prim = z - v_promedio
#traslado del nuevo vector
t_z = np.dot(M,z_prim)
respuesta = traslado(t_z)
