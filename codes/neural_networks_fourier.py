import tensorflow as tf
import math as m
import numpy as np




def ngom_weight_ini(n_input, n_nodes):
    
    weights= []
    W01, W12 = ngom_init(n_input, n_nodes)
    weights.append(W01)
    weights.append(W12)
    print(weights)
   
    return weights

def ngom_init(in_dim, out_dim):
    pi = np.pi
    ngom_stddev1 = 0.6959  #0.13204344312966899640 #np.sqrt((pi * pi/3. - .5)*6./(in_dim * pi*pi)) #np.sqrt((pi*pi/3.0 - .5 - 4.0/(pi*pi))*6/(in_dim * pi*pi))    #np.sqrt(2/(in_dim + out_dim))
    ngom_stddev2 =  np.sqrt(0.512772/ (out_dim * (0.544603))) #np.sqrt(1./(out_dim)) #np.sqrt(1./(3. *out_dim*(1./3.+ 0.3099)))#np.sqrt(1./(out_dim))#np.sqrt(0.505606/(out_dim * (0.506218)))#np.sqrt(pi**2/(3 *out_dim*(pi**2/3 + 0.944871837829830155)))  #np.sqrt((1./3.*pi**2 - 0.83) * 3./(out_dim * pi**2))   #np.sqrt(1./out_dim)  #np.sqrt((pi*pi/3.0 )/(out_dim * (pi*pi/3.0 - 4.0/(pi*pi))))
    w01 = tf.Variable(tf.truncated_normal([in_dim, out_dim], mean = 0.0, stddev=np.sqrt(5.) , dtype=tf.float64), dtype=tf.float64)
    #ngom_stddev2 =np.sqrt(0.5164/ (out_dim * (0.5305)))
    #w01 = tf.Variable(tf.random_uniform([in_dim,out_dim] ,minval = 0.0, maxval = 4.,dtype=tf.float64), dtype=tf.float64)
    ##for Bratu
    #w01 = tf.Variable(tf.random_uniform([in_dim,out_dim] ,minval = 0.0, maxval = 4.,dtype=tf.float64), dtype=tf.float64)
    ###end for bratu
    
    
    w12 = tf.Variable(tf.truncated_normal([out_dim,1], stddev=ngom_stddev2, dtype=tf.float64), dtype=tf.float64)
   
    
    return w01, w12


def ngom_bias_ini(out_dim):
    pi = np.pi
    biases= []
    stddev_bias = pi/np.sqrt(3.0)
    b1 = tf.Variable(tf.zeros([out_dim], dtype=tf.float64), dtype=tf.float64) #tf.Variable(tf.random_uniform([out_dim] ,minval = -pi/2., maxval = pi/2.,dtype=tf.float64), dtype=tf.float64)
    b2 = tf.Variable(tf.zeros([1], dtype=tf.float64), dtype=tf.float64)
    
    biases.append(b1)
    biases.append(b2)
    
    return biases


class neural_network:
    def __init__(self,
                 n_input,
                 n_output,
                 n_hidden_units,
                 n_nodes,
                 activation_hidden=tf.nn.relu,
                 name='velocity_'):
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_units = n_hidden_units
        
        self.activation_hidden = activation_hidden
        self.n_nodes = n_nodes
        
        self.weights = {}
        self.biases = {}
        self.weight_initialization = ngom_weight_ini(self.n_input, self.n_nodes)
        self.bias_initialization = ngom_bias_ini(self.n_nodes)
        
        self.number_of_layers = len(self.n_hidden_units)
        
        
        self.name = name
        
        for i in range(0, self.number_of_layers):
            if i == 0:
                self.weights[self.name+'0'+str(self.n_nodes)] = tf.get_variable(self.name + 'weight_' + str(0)+str(self.n_nodes), initializer=self.weight_initialization[0], dtype=tf.float64)
                #else:
                #self.weights[self.name+str(i)] = tf.get_variable(self.name + 'weight_' + str(i), shape=[self.n_hidden_units[i-1], self.n_hidden_units[i]], initializer=self.weight_initialization, dtype=tf.float64)
            
            self.biases[self.name+str(i)+str(self.n_nodes)] = tf.get_variable(self.name + 'bias_' + str(i)+str(self.n_nodes), initializer=self.bias_initialization[0], dtype=tf.float64)
        
        self.weights[self.name+str(self.number_of_layers)+str(self.n_nodes)] =  tf.get_variable(self.name+'weight_' + str(self.number_of_layers)+str(self.n_nodes), initializer=self.weight_initialization[1], dtype=tf.float64)
        self.biases[self.name+str(self.number_of_layers)+str(self.n_nodes)] =tf.get_variable(self.name+'bias_' + str(self.number_of_layers)+str(self.n_nodes), initializer=self.bias_initialization[1], dtype=tf.float64)

    
                
    def value(self, input_var):
        
        pi = tf.constant(m.pi, dtype = tf.float64)
        layer1 = tf.nn.tanh(tf.add(tf.matmul(input_var, self.weights[self.name+'0'+str(self.n_nodes)]), self.biases[self.name+'0'+str(self.n_nodes)]))
    
    
        layer2 = tf.cos( tf.add(pi*tf.matmul(input_var, self.weights[self.name+'0'+str(self.n_nodes)]), self.biases[self.name+'0'+str(self.n_nodes)]))
        #  layer = tf.concat([layer1, layer2], axis=0)
        
        v1 = tf.matmul(layer1, self.weights[self.name+str(self.number_of_layers)+str(self.n_nodes)]) + self.biases[self.name+str(self.number_of_layers)+str(self.n_nodes)]
        v2 = tf.matmul(layer2, self.weights[self.name+str(self.number_of_layers)+str(self.n_nodes)]) + self.biases[self.name+str(self.number_of_layers)+str(self.n_nodes)]
        #  print(self.biases[self.name+'0'+str(self.n_nodes)].shape)
        # sleep
    
        return v2 #tf.add(v1 ,v2)

        
        
    

    def derivatives(self, X):
        #t = input_var[:,1]
        u = self.value(X)
        print(u.shape)
       
        grad = tf.gradients(u, X)[0]

        grad_grad = []
        
        for i in range(self.n_input):
            grad_grad.append(tf.slice(tf.gradients(tf.slice(grad, [0, i], [tf.shape(X)[0], 1]), X)[0], [0, i],  [tf.shape(X)[0], 1]))
        
        u_x = grad
        u_xx = grad_grad
        return u_x, u_xx
    

