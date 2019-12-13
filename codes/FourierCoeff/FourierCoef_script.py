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
    modes = 5
    for n_nodes in range(1, 16):
        
        N_LAYERS = 1
        BATCHSIZE = 100
        MAX_ITER = 50000
        DO_SAVE = True
        SEED = 42
        
        
        
        HIDDEN_UNITS = []
        for i in range(N_LAYERS):
            HIDDEN_UNITS.append(n_nodes)
        

        problem = obj_fun.objfun()



        NUM_INPUTS = 1
        int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])
        per_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])
        
        neural_network = neural_networks_fourier.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS, n_nodes)
        #neural_network = neural_networks.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS)





        #iny_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])

        
        value_int = neural_network.value(int_var)
        value_per = neural_network.value(per_var)
        # value_bound = neural_network.value(bound_var)
        weight = neural_network.weights
        w1 = list(weight.values())[0]
        w2 = list(weight.values())[1]
        w2 = tf.transpose(w2)
        weight = tf.stack([w1, w2], axis=0)
        regu_weight1 = .0005 * tf.math.reduce_sum(tf.norm(
                              w1,
                              ord='euclidean',
                              axis=None,
                              keepdims=None,
                              name=None
                              ))
        regu_weight2 = .001 * tf.math.reduce_sum(tf.norm(
                                w2,
                                ord='euclidean',
                                axis=None,
                                keepdims=None,
                                name=None
                                ))
        regu_weight = regu_weight1 + regu_weight2
        entier_weight = tf.dtypes.cast(tf.round(w1), tf.float64)
        regu_weight =  .001*tf.math.reduce_sum(tf.norm(
                                                     w1, #- entier_weight,
                                                           ord='euclidean',
                                                           axis=None,
                                                           keepdims=None,
                                                           name=None
                                                           ))
        

        

        
        # sleep
        #grad_grad= neural_network.second_derivatives(int_var)
        
        #grad_grad_sensor = neural_network.second_derivatives(sensor_var)
        
        sol_int = tf.placeholder(tf.float64, [None, 1])
       
        b_value = tf.placeholder(tf.float64, [None, 1])
        
        
        
        loss_int = tf.square(value_int-sol_int)
        loss_per = tf.square(value_int-value_per)
        #loss_bound = tf.square(value_bound - b_value)

        regu_weight = tf.square(regu_weight)
        regu_weight2 = tf.square(regu_weight2)
        
        loss = tf.sqrt(tf.reduce_mean(loss_int + loss_per) + regu_weight + regu_weight2 )# + loss_bound))   #tf.sqrt(regu_weight)
        
        
        train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':MAX_ITER})
        
        
        
        
        
        
        init = tf.global_variables_initializer()
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            
            sess.run(init)
            
            data = scipy.io.loadmat('data/funlog.mat')
            x = data['x'].flatten()[:,None]
            x0 = data['y'].flatten()[:,None]
            #x_per = data['x_per'].flatten()[:,None]
            # n_points = data['N'].flatten()[:,None]
            
           
            int_draw = x#np.concatenate([n_points, t], axis = 1)
            
            
           
           
           
            f = problem.fun(int_draw)
            f = np.reshape(np.array(f), (BATCHSIZE, 1))
            print(f)
            
            # f_b = problem.bound(x0)
            #f_b = np.reshape(np.array(f_b), (BATCHSIZE, 1))
            #print(f_b)
           
           
           
           
           
            
            
            train_scipy.minimize(sess, feed_dict={sol_int:f,  int_var:x, per_var:x0})
            
            
            
            
            
            if DO_SAVE:
                save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(MAX_ITER) + '_rs_' + str(SEED) + '_nodes_' + str(n_nodes)+ '_modes_' + str(modes)
                save_path = saver.save(sess, save_name)
                print("Model saved in path: %s" % save_path)
          


            u_nn = neural_network.value(int_draw)
            ww = neural_network.weights
            bb = neural_network.biases
            w = sess.run(ww)
            b =sess.run(bb)
            print(sess.run(ww))
            print(sess.run(bb))
           
            np.savetxt("nn_fourier.txt", u_nn.eval(), fmt="%s")






if __name__ == '__main__':
    main(sys.argv[1:])
    

