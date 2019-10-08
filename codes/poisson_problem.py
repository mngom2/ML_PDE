import numpy as np

class poisson_2d:
    def __init__(self):
        self.range = [0.0, 1.0]

#solution        
    def velocity(self, x):
        u = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            u[i] = 1.0
            for j in range(2):
                u[i] *= np.sin(np.pi*x[i,j])
        
        return u

    
    def der_velocity(self, x):
        der_u = np.zeros((x.shape[0], 2))
        print(range(x.shape[0]))
        for i in range(x.shape[0]):

            der_u[i,0] = np.pi*np.cos(np.pi*x[i,0]) * np.sin(np.pi*x[i,1])
            der_u[i,1] = np.pi*np.sin(np.pi*x[i,0]) * np.cos(np.pi*x[i,1])
        
        return der_u

    def second_der_velocity(self, x):
        der2_u = np.zeros((x.shape[0], 2))
        
        for i in range(x.shape[0]):
            der2_u[i,0] = -np.pi * np.pi * np.sin(np.pi*x[i,0]) * np.sin(np.pi*x[i,1])

            der2_u[i,1] = -np.pi * np.pi * np.sin(np.pi*x[i,0]) * np.sin(np.pi*x[i,1])
        
        return der2_u


    def rhs(self, x):
        f = np.zeros((x.shape[0], 1))
        # f2 = np.zeros((x.shape[0], 1))
    
        for i in range(x.shape[0]):
            f[i] = 2.0*np.pi*np.pi  * np.sin(np.pi * x[i,0]) * np.sin(np.pi * x[i,1]) *  x[i,0]   -1.0*np.pi* np.cos( np.pi * x[i,0] )* np.sin( np.pi * x[i,1] ) #* x[i,0]
        #f2[i] = -1.0*np.pi* np.cos(np.pi * x[i,0])* np.sin(np.pi * x[i,1])
        #   for j in range(2):
        #  f1[i] *= x[i,j] * np.sin(np.pi*x[i,j])
                    #             if(j == 0):
                    #f2[i] *= x[i,j] * np.cos(np.pi*x[i,j])
                    #else:
                    #f2[i] *= np.sin(np.pi*x[i,j])
        
        
        return f
