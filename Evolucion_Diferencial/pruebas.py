from Evolucion_Diferencial.algorithm_DE import DE
from Evolucion_Diferencial.MLP_DE import MLP_DE

import matplotlib.pyplot as plt

mlp_de = MLP_DE(4,3)
indices = mlp_de.calcula_indices()
"""
inicializar algortimo de evolución diferencial
@params
Np      = Número de individuos
Dim     = Dimensionalidad de la poblacion
Cr      = Operador de cruzamiento (Crossover Ratio)
F       = Factor de Mutación
"""
#de = DE(Np=20,Dim=5,Cr=0.8,F=0.9)
"""
generando poblacion inicial
@params
LI = Límite inferior
LS = Límite superior
@variable pob_(name_function) == población inicial
"""
#pob_sphere = de.generate_initial_pob(LI=-5.12,LS=5.12)
#pob_ackley = de.generate_initial_pob(LI=-32.768,LS=32.768)
#pob_rosenbrock = de.generate_initial_pob(LI=-2.048,LS=2.048)
#pob_sigmoid = de.generate_initial_pob(LI=-1.0,LS=1.0)
#pob_square_error = de.generate_initial_pob(LI=-1.0,LS=1.0)

"""
resultados
@params
pob_(name_function) = población inicial
epochs              = generaciones
obj_fun = función objetivo (sphere,ackley,rosenbrock)
"""
#v_result = de.DE_function(pob_init=pob_sigmoid,epochs=100,obj_fun="square_error")










