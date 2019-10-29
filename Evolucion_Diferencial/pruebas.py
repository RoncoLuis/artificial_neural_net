from Evolucion_Diferencial.algorithm_DE import DE
import matplotlib.pyplot as plt
"""
inicializar algortimo de evolución diferencial
@params
Np      = Número de individuos
Dim     = Dimensionalidad de la poblacion
Cr      = Operador de cruzamiento (Crossover Ratio)
F       = Factor de Mutación
"""
de = DE(Np=20,Dim=5,Cr=0.8,F=0.9)
"""
generando poblacion inicial
@params
LI = Límite inferior
LS = Límite superior
@variable pob_(name_function) == población inicial
"""
pob_sphere = de.generate_initial_pob(LI=-5.12,LS=5.12)
pob_ackley = de.generate_initial_pob(LI=-32.768,LS=32.768)
pob_rosenbrock = de.generate_initial_pob(LI=-2.048,LS=2.048)

"""
resultados
@params
pob_(name_function) = población inicial
epochs              = generaciones
obj_fun = función objetivo (sphere,ackley,rosenbrock)
"""
v_result = de.DE_function(pob_init=pob_sphere,epochs=10000,obj_fun="sphere")










