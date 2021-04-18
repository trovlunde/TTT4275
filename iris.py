"""TTT4275 Project in Classification"""
import numpy as np
import matplotlib.pyplot as plt

CLASSES = 3
legends = ['Setosa', 'Versicolour', 'Virginica']

def load_Iris_data():
    for i in range(CLASSES):
        tmp = np.loadtxt("./Iris_TTT4275/class_"+str(i+1),delimiter=",")
        class_number = np.ones((tmp.shape[0],2))
        class_number[:,-1] *= i
        tmp = np.hstack((tmp, class_number))
        if i > 0:
            data = np.vstack((data,tmp))
        else:
            data = copy.deepcopy(tmp)
    tmp = data[:,:-1]
    tmp = tmp / tmp.max(axis=0)
    data[:,:-1] = tmp
    return data

print(load_Iris_data)
