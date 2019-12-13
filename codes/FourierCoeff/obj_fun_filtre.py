import numpy as np

class objfun:
    def __init__(self):
        self.range = [-1.0, 1.0]
    
    def fun(self, x):
        u = np.zeros((x.shape[0], 1))
        v = np.zeros((x.shape[0], 1))
        w = np.zeros((x.shape[0], 1))
        #u = np.fft.ifft(np.fft.fft(np.exp(-np.abs(x))))
        for i in range(x.shape[0]):
            u[i] = np.cos(10.*(x[i,0])) #np.cos(10. *x[i,0])
            v[i] =x[i,0]**2
            w[i] = x[i,0]**3
            #u[i] = (1. - np.exp(-np.pi))/np.pi + 2./np.pi * (1./(1. + 1.)) * (1. + np.exp(-np.pi)) *  np.cos(x[i,0]) +  2./np.pi * (1./(1. + 4.)) * (1. - np.exp(-np.pi)) *  np.cos(2. *x[i,0]) + 2./np.pi * (1./(1. + 9.)) * (1. + np.exp(-np.pi)) *  np.cos(3. *x[i,0]) +2./np.pi * (1./(1. + 16.)) * (1. - np.exp(-np.pi)) *  np.cos(4. *x[i,0])
        #u = np.fft.ifft(np.fft.fft(np.exp(-np.abs(x))))
        
        
        #(1 - np.exp(-np.pi))/np.pi + 2/np.pi * (1/(1 + 1)) * (1 + np.exp(-np.pi)) *  np.cos(x[i,0]) #np.exp(-np.abs(x[i,0])) #np.cos(x[i,0])  #
         #np.sin(x[i,0])*np.sin(x[i,0])*np.sin(x[i,0]) #np.exp(x[i,0])  #np.sin(x[i,0])*np.sin(x[i,0])*np.sin(x[i,0]) #*x[i,0]
        return np.concatenate([np.transpose(u),np.transpose(v),np.transpose(w)], axis = 1)
    
    def rhs(self, x):
        f = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            f[i] =100.0 * 100.0 * np.sin(100.0*x[i,0])
        return f
