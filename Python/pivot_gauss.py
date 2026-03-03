import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from copy import deepcopy

def Gauss(M):
    n = M.shape[0]
    for j in range(min(M.shape[1] - 1, n)):
        # Pivot partiel
        max_row = j
        for k in range(j + 1, n):
            if abs(M[k, j]) > abs(M[max_row, j]):
                max_row = k
        M[[j, max_row]] = M[[max_row, j]]

        # Éliminer les éléments sous le pivot
        for i in range(j + 1, n):
            factor = M[i, j] / M[j, j]
            M[i, j:] -= factor * M[j, j:]

    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = (M[i, -1] - np.dot(M[i, i+1:-1], X[i+1:])) / M[i, i]

if __name__ == "__main__":	
    n = 10 #samples
    X = np.arange(n)
    Y = 20*(np.random.rand(n)-0.5) + np.random.randint(-5, 5)*X + 1    
    A = np.array([[xi, 1] for xi in X])
    A1 = A[:, 0]
    A2 = A[:, 1]
    Matrice_augmentee = np.column_stack((A1, A2, Y))

    print(Matrice_augmentee)


    
    #Matrice 3x3 de demonstration
    Matrice_augmentee = np.array([[ 2.0, -1.0,  0.0],
                                 [-1.0,  2.0, -1.0],
                                 [ 0.0, -1.0,  2.0]])
    
    print(Gauss(Matrice_augmentee))