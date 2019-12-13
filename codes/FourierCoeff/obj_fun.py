import numpy as np

class objfun:
    def __init__(self):
        self.range = [-np.pi, np.pi]
    
    def fun(self, x): #,N):
        u = np.zeros((x.shape[0],1))
        v = np.zeros((x.shape[0], 1))
        w = np.zeros((x.shape[0], 1))
        N = 18
        #u = np.fft.ifft(np.fft.fft(np.exp(-np.abs(x))))
        for i in range(x.shape[0]):
            w[i] = np.cos(10.*  x[i,0]) + np.sin(7. * x[i,0])  + np.sin(2. * x[i,0]) #+ np.cos(10. * x[i,0])
            if (x[i,0] > np.pi):
                v[i] = ((x[i,0]) - 2.*np.pi)**2
            elif (x[i,0] > 3.*np.pi):
                v[i] = ((x[i,0]) - 4.*np.pi)**2
            else:
                v[i] = ((x[i,0]))**2 #np.cos(10. * x[i,0]) + np.sin(7. * x[i,0]) #(x[i,0])**2 #np.cos(x[i,0]) + np.sin(x[i,0])  #(x[i,0]) # np.cos(x[i,0]) + np.sin(x[i,0]) #np.cos(x[i,0])  +  np.sin(x[i,0]) #(x[i,0])**2   #np.sin(30.0 * x[i,0])
                #if (-np.pi <= x[i,0] < 0.0):
                #u[i] = -1.0 #np.exp(- np.abs(x[i,0]))    #np.cos(x[i,0])
                #else:
                #u[i] = 1.0
        
            
            #u[i] = (np.pi)**2/3
            #for j in range(N):
            #   v[i] = x[i,0]
            #   print (j)
            #u[i] = u[i] + (-1.)**(j+1) * -2. *np.sin(((j+1)*x[i,0]))/((j+1)) #np.cos(10. *x[i,0])
            #v[i] =x[i,0]**2
            #w[i] = x[i,0]**3
            #u[i] = (1. - np.exp(-np.pi))/np.pi + 2./np.pi * (1./(1. + 1.)) * (1. + np.exp(-np.pi)) *  np.cos(x[i,0]) +  2./np.pi * (1./(1. + 4.)) * (1. - np.exp(-np.pi)) *  np.cos(2. *x[i,0]) + 2./np.pi * (1./(1. + 9.)) * (1. + np.exp(-np.pi)) *  np.cos(3. *x[i,0]) +2./np.pi * (1./(1. + 16.)) * (1. - np.exp(-np.pi)) *  np.cos(4. *x[i,0])
        #u = np.fft.ifft(np.fft.fft(np.exp(-np.abs(x))))
        
        
        #(1 - np.exp(-np.pi))/np.pi + 2/np.pi * (1/(1 + 1)) * (1 + np.exp(-np.pi)) *  np.cos(x[i,0]) #np.exp(-np.abs(x[i,0])) #np.cos(x[i,0])  #
         #np.sin(x[i,0])*np.sin(x[i,0])*np.sin(x[i,0]) #np.exp(x[i,0])  #np.sin(x[i,0])*np.sin(x[i,0])*np.sin(x[i,0]) #*x[i,0]
        
         
        return v #u#, v #np.concatenate([np.transpose(u),np.transpose(v),np.transpose(w)], axis = 1)
    
    def rhs(self, x):
        f = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            f[i] =100.0 * 100.0 * np.sin(100.0*x[i,0])
        return f
    def bound(self, x):
        f = np.zeros((x.shape[0], 1))
    
        for i in range(x.shape[0]):
            f[i] = (np.pi)**2;
        return f
