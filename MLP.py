import numpy as np
from activations import *

class DenseNetwork:
    def __init__(self, layers_dim, hidden_activation=tanh, output_activation=logistic):
        #Attributes
        self.L = len(layers_dim) - 1
        self.w = [None] * (self.L + 1)
        self.b = [None] * (self.L + 1)
        self.f = [None] * (self.L + 1)

        #Inicialize weights and biases
        for l in range(1, self.L + 1):
            self.w[l] = -1 + 2 * np.random.rand(layers_dim[l], layers_dim[l-1])
            self.b[l] = -1 + 2 * np.random.rand(layers_dim[l], 1)

            if l == self.L:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation
        
    def predict(self, X):
        a = X
        for l in range(1, self.L + 1):
            z = self.w[l] @ a + self.b[l]
            a = self.f[l](z)
        return a
    
    def fit(self, X, Y, epochs=500, lr=0.1):
        p = X.shape[1]

        for _ in range(epochs):
            #Inicialize activations and gradients
            a = [None] * (self.L + 1)
            da = [None] * (self.L + 1)
            lg = [None] * (self.L + 1)

            #Propagation
            a[0] = X
            for l in range(1, self.L + 1):
                z = self.w[l] @ a[l - 1] + self.b[l]
                a[l], da[l] = self.f[l](z, derivative=True)
            
            #Backpropagation
            for l in range(self.L, 0, -1):
                if l == self.L:
                    lg[l] = - (Y - a[l]) * da[l]
                else:
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * da[l]
            
            #Gradient Descent
            for l in range(l, self.L + 1):
                self.w[l] -= (lr/p) * (lg[l] @ a[l-1].T)
                self.b[l] -= (lr/p) * np.sum(lg[l])
