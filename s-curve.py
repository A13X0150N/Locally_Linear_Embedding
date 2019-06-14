# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:11:10 2019

@author: Ryan
"""
import matplotlib.cm as cm
import gzip
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
import scipy as sp
from sklearn import manifold
from sklearn import neighbors
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import struct



#
# Inputs: Y    --> Sample matrix
#         n, m --> Landmark matrix dimensions
#
# Output: idx  --> nxm Landmark matrix
#
def find_landmarks(Y, n, m):
    xr = np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), n)
    yr = np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), m)
    xg, yg = np.meshgrid(xr, yr)
    idx = [0]*(n*m)
    for i, x, y in zip(range(n*m), xg.flatten(), yg.flatten()):
        idx[i] = int(np.sum(np.abs(Y-np.array([x,y]))**2, axis=-1).argmin())
    return idx






X, t = datasets.make_s_curve(2500)
print("X:", np.shape(X))



# Train algorithm and calculate landmark graph
Y, err = manifold.locally_linear_embedding(X.T, n_neighbors=10, n_components=2)
#Y, err = locally_linear_embedding(X, n_neighbors=10, n_components=2)
landmarks = find_landmarks(Y, 5, 5)

# Plot the clustered data with landmarks overlaid
plt.scatter(Y[:,0], Y[:,1])
plt.scatter(Y[landmarks,0], Y[landmarks,1])

# Show the landmark samples in a 5x5 grid
fig = plt.figure(figsize=(15,15))
for i in range(len(landmarks)):
    ax = fig.add_subplot(5, 5, i+1)
    imgplot = ax.imshow(Y, cmap=plt.cm.get_cmap("Greys"))
    imgplot.set_interpolation("nearest")
plt.show()




#fig2 = plt.figure(figsize=(10,10))
#ax1 = fig2.add_subplot(111, projection='3d')
#ax1.scatter(Xt[0,:], Xt[1,:], Xt[2,:], marker='o')

#plt.legend()
#plt.show()









