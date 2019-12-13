import tensorflow as tf
tf.set_random_seed(42)

import import_ipynb
import numpy as np
from scipy import integrate
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import scipy.io
import neural_networks
import neural_networks_fourier
import obj_fun
import matplotlib.pyplot as plt
import sys, getopt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math as m



def main(argv):
    
    # DEFAULt
    n_nodes = 100
    N_LAYERS = 3
    BATCHSIZE = 101
    MAX_ITER = 50000
    DO_SAVE = True
    SEED = 42
    
    
    
    HIDDEN_UNITS = []
    for i in range(N_LAYERS):
        HIDDEN_UNITS.append(n_nodes)
    

    problem = obj_fun.objfun()



    NUM_INPUTS = 1
    int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])
    #neural_network = neural_networks_fourier.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS, n_nodes)
    neural_network = neural_networks.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS)





    #iny_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])

    
    
    

    

    
    # sleep
    grad= neural_network.first_derivatives(int_var)
    norm_grad = tf.math.reduce_sum(tf.norm(
                                           grad,
                                           ord='euclidean',
                                           axis=None,
                                           keepdims=None,
                                           name=None
                                           ))
    
    #grad_grad_sensor = neural_network.second_derivatives(sensor_var)
    
    # sol_int = tf.placeholder(tf.float64, [None, 1])

    
    
    
    loss_int = (1+ tf.multiply(norm_grad, norm_grad))
  
    #regu_weight = tf.square(regu_weight)
    
    loss = tf.sqrt(tf.reduce_mean(loss_int ))   #tf.sqrt(regu_weight)
    
    
    train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':MAX_ITER})
    
    
    
    
    
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        
        sess.run(init)
        
        data = scipy.io.loadmat('data/funexp.mat')
        x = data['x'].flatten()[:,None]
        # n_points = data['N'].flatten()[:,None]
        
       
        int_draw = x#np.concatenate([n_points, t], axis = 1)
        print(int_draw.shape)
        
       
        
        
       
       
        
        
       
        
        
        
        train_scipy.minimize(sess, feed_dict={int_var:int_draw})
        
        
        
        
        
        if DO_SAVE:
            save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(MAX_ITER) + '_rs_' + str(SEED)
            save_path = saver.save(sess, save_name)
            print("Model saved in path: %s" % save_path)


        u_nn = neural_network.value(int_draw)
        ww = neural_network.weights
        bb = neural_network.biases
        print(sess.run(ww))
        print(sess.run(bb))
        np.savetxt("nn_fourier.txt", u_nn.eval(), fmt="%s")





if __name__ == '__main__':
    main(sys.argv[1:])
    

