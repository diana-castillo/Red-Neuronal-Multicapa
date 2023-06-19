import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLP import *

def MLP_binary_classification_2d(X,Y,net):
    plt.figure()
    for i in range(X.shape[1]):
        if Y[0,i]==0:
            plt.plot(X[0,i], X[1,i], 'ro', markersize=9)
        else:
            plt.plot(X[0,i], X[1,i], 'bo', markersize=9)
    xmin, ymin=np.min(X[0,:])-0.5, np.min(X[1,:])-0.5
    xmax, ymax=np.max(X[0,:])+0.5, np.max(X[1,:])+0.5
    xx, yy = np.meshgrid(np.linspace(xmin,xmax, 100), 
                         np.linspace(ymin,ymax, 100))
    data = [xx.ravel(), yy.ravel()]
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    plt.contour(xx,yy,zz,[0.5], colors='k',  linestyles='--', linewidths=2)
    plt.contourf(xx,yy,zz, alpha=0.8, 
                 cmap=plt.cm.RdBu)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.grid()
    plt.show()
    

opcion = 1

if opcion == 1:
    csv = 'blobs.csv'
elif opcion == 2:
    csv = 'circles.csv'
elif opcion == 3:
    csv = 'moons.csv'
else:
    csv = 'XOR.csv'

df = pd.read_csv(csv)
X = np.asanyarray(df.drop(columns=['y'])).T
Y = np.asanyarray(df[['y']]).T

net = DenseNetwork((2,100,1))
#print(net.predict(X))
MLP_binary_classification_2d(X,Y,net)

net.fit(X, Y)
#print(net.predict(X))
MLP_binary_classification_2d(X,Y,net)