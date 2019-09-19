"""
Single-Layer Net -> red de capa simple
"""
x1 = [1,1,0,0]
x2 = [1,0,1,0]
w1 = 1
w2 = 1
umbral = 2

for i in x1:
    sum = (x1[1]*w1)+(x2[i]*w2)
    print("sumatoria: ",sum)
    if sum > umbral:
        print("se dispara neurona ->",1,":",i)
    elif sum <= umbral:
        print("no se disparo neurona ->",0,":",i)
    else:
        print("Entro aqui")