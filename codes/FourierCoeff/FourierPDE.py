import tensorflow as tf
tf.set_random_seed(42)

import import_ipynb
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import scipy.io
import neural_networks
import neural_networks_fourier
import poisson_problem
import matplotlib.pyplot as plt
import sys, getopt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math as m



def main(argv):
    
    # DEFAULT
    n_nodes = 20
    N_LAYERS = 1
    BATCHSIZE = 100
    MAX_ITER = 50000
    DO_SAVE = True
    SEED = 42
    
    
    
    HIDDEN_UNITS = []
    for i in range(N_LAYERS):
        HIDDEN_UNITS.append(n_nodes)
    
    
    problem = poisson_problem.poisson_1d()



    NUM_INPUTS = 1
    int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])
    neural_network = neural_networks_fourier.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS, n_nodes)
    #neural_network = neural_networks.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS, n_nodes)




    #ic_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])
    #iny_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])

    
    value_int = neural_network.value(int_var)
    value_per = neural_network.value(int_p)
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
                                                  w1, # - entier_weight,
                                       ord='euclidean',
                                      axis=None,
                                       keepdims=None,
                                      name=None
                                      ))
    
    #value_ic = neural_network.value(ic_var)
    
    
    u_x, u_xx= neural_network.derivatives(int_var)
    print(u_xx.shape)
    
    # sleep
    #grad_grad= neural_network.second_derivatives(int_var)
    
    #grad_grad_sensor = neural_network.second_derivatives(sensor_var)
    
    sol_int = tf.placeholder(tf.float64, [None, 1])
    # sol_ic = tf.placeholder(tf.float64, [None, 1])
    
    
    
    loss_int = tf.square(u_xx+ sol_int)
    #loss_ic =  tf.square(value_ic-sol_ic)
    #regu_weight = tf.square(regu_weight1) + tf.square(regu_weight2)
    regu_weight = tf.square(regu_weight)
    regu_weight2 = tf.square(regu_weight2)
    
    loss = tf.sqrt(tf.reduce_mean(loss_int) + regu_weight + regu_weight2 )
    
    #loss = tf.sqrt(tf.reduce_mean(loss_int + loss_ic + regu_weight ))
    
    
    train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='SQSLP', options={'gtol':1e-14, 'disp':True, 'maxiter':MAX_ITER})
    
    
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        
        sess.run(init)
        
        data = scipy.io.loadmat('data/funlog_per.mat')
        x = data['x'].flatten()[:,None]
        # n_points = data['N'].flatten()[:,None]
        x_per = data['x_per'].flatten()[:,None]
       
        int_draw = x_per#np.concatenate([n_points, t], axis = 1)
        print(int_draw.shape)
        
        #ic_draw = x0  #np.concatenate([n_points, t0], axis = 1)
        #ic_draw = ic_draw.astype(np.float64)
        
        #        boundary_draw_x, boundary_draw_y = sampler.boundary_samples(BATCHSIZE)
        #        boundary_draw_x = np.reshape(boundary_draw_x, (BATCHSIZE, 1))
        #        boundary_draw_y = np.reshape(boundary_draw_y, (BATCHSIZE, 1))
        
        
        
        #        bou_draw = np.concatenate([boundary_draw_x, boundary_draw_y], axis=1)
        
        
        f = problem.rhs(x)
        f = np.reshape(np.array(f), (BATCHSIZE, 1))
        print(f.shape)
        
        
        #ic = problem.velocity(ic_draw)
        #ic = np.reshape(np.array(ic), (BATCHSIZE, 1))
      
        #print(ic)
        #        for i in range(ic.shape[0]):
        # print(ic[i,0])
        #sleep
        
        
        train_scipy.minimize(sess, feed_dict={sol_int:f,  int_var:int_draw})
        #train_scipy.minimize(sess, feed_dict={sol_int:f,  int_var:int_draw, sol_ic: ic, ic_var: ic_draw})
        
        
        
        
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
    

