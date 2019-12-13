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
    ngom_stddev1 =  np.sqrt((pi * pi/3. - .5)*6./(in_dim * pi*pi)) #np.sqrt((pi*pi/3.0 - .5 - 4.0/(pi*pi))*6/(in_dim * pi*pi))    #np.sqrt(2/(in_dim + out_dim))
    ngom_stddev2 = np.sqrt(1./out_dim)  #np.sqrt((pi*pi/3.0 )/(out_dim * (pi*pi/3.0 - 4.0/(pi*pi))))
    #w01 = tf.Variable(tf.truncated_normal([in_dim, out_dim], mean = 0.0, stddev=ngom_stddev1, dtype=tf.float64), dtype=tf.float64)
    w01 = tf.Variable(tf.random_uniform([in_dim,out_dim] ,minval = 0.0, maxval = 20.0,dtype=tf.float64), dtype=tf.float64)
    
    
    w12 = tf.Variable(tf.truncated_normal([out_dim,1], stddev=ngom_stddev2, dtype=tf.float64), dtype=tf.float64)
   
    
    return w01, w12


def ngom_bias_ini(out_dim):
    pi = np.pi
    biases= []
    stddev_bias = pi/np.sqrt(3.0)
    b1 = tf.Variable(tf.random_uniform([out_dim] ,minval = -pi, maxval = pi,dtype=tf.float64), dtype=tf.float64)
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

    
    def sin_activation(self,x):
        pi = tf.constant(m.pi, dtype = tf.float64)
        zero = tf.constant(0.0, dtype = tf.float64)
        one = tf.constant(1.0, dtype = tf.float64)
        part_1 = tf.dtypes.cast(tf.to_float(tf.math.less(x, -pi/2.0)), dtype = tf.float64)
        
        
        part_2 = tf.dtypes.cast(tf.to_float(tf.math.logical_and(tf.math.less_equal(x, pi/2.0), tf.math.less_equal(-pi/2.0, x))), dtype = tf.float64)
        
        
        part_3 = tf.dtypes.cast(tf.to_float(tf.math.less(pi/2.0, x)), dtype = tf.float64)
        return part_1*tf.nn.sigmoid(x)+ part_2*tf.nn.sigmoid(x) + part_3 * tf.nn.sigmoid(x)
                    
                    
                    
    def cos_activation(self,x):
        pi = tf.constant(m.pi, dtype = tf.float64)
        zero = tf.constant(0.0, dtype = tf.float64)
        one = tf.constant(1.0, dtype = tf.float64)
        part_1 = tf.dtypes.cast(tf.to_float(tf.math.less(x, zero)), dtype = tf.float64)
        
        
        part_2 = tf.dtypes.cast(tf.to_float(tf.math.logical_and(tf.math.less_equal(x, pi), tf.math.less_equal(zero, x))), dtype = tf.float64)
        
        
        part_3 = tf.dtypes.cast(tf.to_float(tf.math.less(pi, x)), dtype = tf.float64)
        return part_1*0.0 + part_2*tf.cos(x) + part_3 * 1.0
                
    def value(self, input_var):
        
        pi = tf.constant(m.pi, dtype = tf.float64)
        layer1 = tf.nn.tanh(tf.add(tf.matmul(input_var, self.weights[self.name+'0'+str(self.n_nodes)]), self.biases[self.name+'0'+str(self.n_nodes)]))
    
    
        layer2 = tf.cos(tf.add(tf.matmul(input_var, self.weights[self.name+'0'+str(self.n_nodes)]), self.biases[self.name+'0'+str(self.n_nodes)]))
        #  layer = tf.concat([layer1, layer2], axis=0)
        
        v1 = tf.matmul(layer1, self.weights[self.name+str(self.number_of_layers)+str(self.n_nodes)]) + self.biases[self.name+str(self.number_of_layers)+str(self.n_nodes)]
        v2 = tf.matmul(layer2, self.weights[self.name+str(self.number_of_layers)+str(self.n_nodes)]) + self.biases[self.name+str(self.number_of_layers)+str(self.n_nodes)]
        print(self.biases[self.name+'0'+str(self.n_nodes)].shape)
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
        
        u_x = grad[0]
        u_xx = grad_grad[0]
        return u_x, u_xx
    

