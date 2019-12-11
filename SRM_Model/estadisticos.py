import numpy as np
import matplotlib.pyplot as plt

acc_300_gen = np.genfromtxt('acc_100gen.csv',delimiter=',')

x = ['entrenamiento','prueba','promedio']
y = [acc_300_gen[:,0],acc_300_gen[:,1],acc_300_gen[:,2]]

acc_result = np.array(accs)
promedios = np.mean(acc_result,axis=1)
acc_result = np.column_stack([acc_result,promedios])
np.savetxt('acc_1gen.csv',acc_result,delimiter=',')

plt.title('SRM 10 gen iris')
plt.ylabel('accuracy')
plt.xticks([1, 2, 3], ['entrenamiento', 'prueba', 'promedio'])
plt.boxplot(acc_result)
plt.show()
