import tensorflow as tf
import math as m





class neural_network:
    def __init__(self,
                 n_input,
                 n_output,
                 n_hidden_units,
                 n_nodes,
                 weight_initialization=tf.contrib.layers.xavier_initializer(), #tf.contrib.layers.xavier_initializer(),
                 activation_hidden=tf.nn.relu,
                 name='velocity_'):
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_units = n_hidden_units
        self.weight_initialization = weight_initialization
        self.activation_hidden = activation_hidden
        self.n_nodes = n_nodes
        
        self.weights = {}
        self.biases = {}
        self.number_of_layers = len(self.n_hidden_units)
        
        
        
        self.name = name
        
        for i in range(0, self.number_of_layers):
            if i == 0:
                self.weights[self.name+'0'] = tf.get_variable(self.name + 'weight_' + str(0), shape=[self.n_input, self.n_hidden_units[0]], initializer=self.weight_initialization, dtype=tf.float64)
            
            else:
                self.weights[self.name+str(i)] = tf.get_variable(self.name + 'weight_' + str(i), shape=[self.n_hidden_units[i-1], self.n_hidden_units[i]], initializer=self.weight_initialization, dtype=tf.float64)
            
            self.biases[self.name+str(i)] = tf.get_variable(self.name + 'bias_' + str(i), initializer= tf.dtypes.cast(tf.zeros([self.n_hidden_units[i]]), dtype = tf.float64), dtype=tf.float64)
        
        self.weights[self.name+str(self.number_of_layers)] =  tf.get_variable(self.name+'weight_' + str(self.number_of_layers), shape=[self.n_hidden_units[-1], self.n_output], initializer=self.weight_initialization, dtype=tf.float64)
        self.biases[self.name+str(self.number_of_layers)] =tf.get_variable(self.name+'bias_' + str(self.number_of_layers), initializer= tf.dtypes.cast(tf.zeros([self.n_output]), dtype=tf.float64),dtype=tf.float64)

    
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
        
            
        layer1 = tf.nn.tanh(tf.add(tf.matmul(input_var, self.weights[self.name+'0']), self.biases[self.name+'0']))
    
    
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(input_var, self.weights[self.name+'0']), self.biases[self.name+'0']))
        #  layer = tf.concat([layer1, layer2], axis=0)
        
        v1 = tf.matmul(layer1, self.weights[self.name+str(self.number_of_layers)]) + self.biases[self.name+str(self.number_of_layers)]
        v2 = tf.matmul(layer2, self.weights[self.name+str(self.number_of_layers)]) + self.biases[self.name+str(self.number_of_layers)]
        return v2 #tf.add(v1 ,v2)

        
        
    

    def derivatives(self, X):
        #t = input_var[:,1]
        u = self.value(X)
       
        grad = tf.gradients(u, X)[0]

        grad_grad = []
        
        for i in range(self.n_input):
            grad_grad.append(tf.slice(tf.gradients(tf.slice(grad, [0, i], [tf.shape(X)[0], 1]), X)[0], [0, i],  [tf.shape(X)[0], 1]))
        
        u_x = grad[0]
        u_xx = grad_grad[0]
        return u_x, u_xx
    

