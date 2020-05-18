import numpy as np

class objfun:
    def __init__(self):
        self.range = [-1., 1.]
    
    def fun(self, x): #,N):
        #u = np.zeros((x.shape[0],1))
        #xsquare = np.zeros((x.shape[0], 1))
        #w = np.zeros((x.shape[0], 1))
        absx = np.zeros((x.shape[0], 1))
        #u = np.fft.ifft(np.fft.fft(np.exp(-np.abs(x))))
        for i in range(x.shape[0]):
            absv[i] = np.abs(x[i,0])
            #xsquare = (x[i,0])**2
            #u[i] = np.cos(np.pi*x[i,0]) + np.sin(np.pi *x[i,0])
            # w[i] = 8. * np.cos( 4. *np.pi*x[i,0]) + np.sin(2. * np.pi *x[i,0]) + np.sin(np.pi *x[i,0])
    
        
         
        return absx #xsquare #w #u
    
