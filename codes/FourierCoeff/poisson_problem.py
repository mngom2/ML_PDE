import numpy as np

class poisson_1d:
    def __init__(self):
        self.range = [-np.pi, 3*np.pi]
    
    def velocity(self, x):
        u = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            u[i] =  np.cos(np.pi * x[i,0])  #x[i,0]**2 #np.exp(-(x[i,0]**2))
        return u
    
    def rhs(self, x):
        f = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            f[i] = np.pi * np.pi * np.cos(np.pi*x[i,0])  #-2.0 #(2  - 4 * x[i,0]**2) * np.exp(-(x[i,0]**2)) #  -np.exp(x[i,0])
        return f
