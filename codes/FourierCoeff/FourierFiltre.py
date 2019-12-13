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
    BATCHSIZE = 1000
    MAX_ITER = 50000
    DO_SAVE = True
    SEED = 42
    n_train = 3
    loss_int = 0.0
    
    
    HIDDEN_UNITS = []
    for i in range(N_LAYERS):
        HIDDEN_UNITS.append(n_nodes)
    

    problem = obj_fun.objfun()



    NUM_INPUTS = 2
    int_var = tf.placeholder(tf.float64, [None, n_train * NUM_INPUTS])
    #neural_network = neural_networks_fourier.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS, n_nodes)
    neural_network = neural_networks.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS)
    sol_int = tf.placeholder(tf.float64, [None, n_train])





    #iny_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])
    for i in range(n_train):
        
        value_int = neural_network.value(int_var[:,i*(NUM_INPUTS):(i*NUM_INPUTS) + NUM_INPUTS])
        
 
    
    

    
    
    
        loss_int = loss_int + tf.square(value_int-sol_int[:,i])

    #regu_weight = tf.square(regu_weight)
    
    loss = tf.sqrt(tf.reduce_mean(loss_int  ))   #tf.sqrt(regu_weight)
    
    
    train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':MAX_ITER})
    
    
    
    
    
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        
        sess.run(init)
        
        data = scipy.io.loadmat('data/filtre_train.mat')
        ff1 = data['ff1'].flatten()[:,None]
        ff2 = data['ff2'].flatten()[:,None]
        ff3 = data['ff3'].flatten()[:,None]
        
        x_points = data['x'].flatten()[:,None]
        int_draw_X, int_draw_Y1 = np.meshgrid(x_points,ff1)
        int_draw_X, int_draw_Y2 = np.meshgrid(x_points,ff2)
        int_draw_X, int_draw_Y3 = np.meshgrid(x_points,ff3)
        #int_draw_X, int_draw_Y1 = np.meshgrid(x_points,ff)
        int_draw_x = int_draw_X.flatten()[:, None]
        int_draw_y1 = int_draw_Y1.flatten()[:, None]
        int_draw_y2 = int_draw_Y2.flatten()[:, None]
        int_draw_y3 = int_draw_Y3.flatten()[:, None]
        print(int_draw_y2.shape)
        int_draw = np.concatenate([int_draw_x, int_draw_y1, int_draw_x, int_draw_y2,int_draw_x, int_draw_y3,], axis=1)
        
       
       #int_draw = np.concatenate([x_points, ff], axis = 1)
        print(int_draw)
        
        
       
        
        
        f = problem.fun(int_draw_x)
        
        f = np.reshape(np.array(f), (BATCHSIZE, 3))
       
        
        
       
        
        
        
        train_scipy.minimize(sess, feed_dict={sol_int:f,  int_var:int_draw})
        
        
        
        
        
        if DO_SAVE:
            save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(MAX_ITER) + '_rs_' + str(SEED)
            save_path = saver.save(sess, save_name)
            print("Model saved in path: %s" % save_path)


        #u_nn = neural_network.value(int_draw)
        #ww = neural_network.weights
        #bb = neural_network.biases
        #print(sess.run(ww))
        #print(sess.run(bb))
#np.savetxt("nn_fourier.txt", u_nn.eval(), fmt="%s")





if __name__ == '__main__':
    main(sys.argv[1:])
    


#weight = neural_network.weights
#w1 = list(weight.values())[0]
#w2 = list(weight.values())[1]
#w2 = tf.transpose(w2)
#weight = tf.stack([w1, w2], axis=0)
#regu_weight1 = .0005 * tf.math.reduce_sum(tf.norm(
#                     w1,
#                      ord='euclidean',
#                      axis=None,
#                      keepdims=None,
#                      name=None
#                      ))
#regu_weight2 = .00005 * tf.math.reduce_sum(tf.norm(
#                        w1,ord='euclidean',
#                       axis=None,
#                        keepdims=None,
#                        name=None
#                        ))
# regu_weight = regu_weight1 + regu_weight2
# regu_weight = 0.000005 * tf.math.reduce_sum(tf.norm(
#                                                   weight,
#                                                   ord='euclidean',
#                                                   axis=None,
#                                                   keepdims=None,
#                                                  name=None
#                                                 ))





# sleep
#grad_grad= neural_network.second_derivatives(int_var)

#grad_grad_sensor = neural_network.second_derivatives(sensor_var)
