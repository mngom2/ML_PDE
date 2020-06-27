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
    
    
    
    
    def print_loss(loss_evaled):
       
        print(loss_evaled)
    
    #np.savetxt("loss_fourier.txt", loss_evaled, fmt="%s")
    
    # DEFAULt
    n_nodes = 4
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
    per_var1 = tf.placeholder(tf.float64, [None, NUM_INPUTS])
    per_var2 = tf.placeholder(tf.float64, [None, NUM_INPUTS])
    per_var3 = tf.placeholder(tf.float64, [None, NUM_INPUTS])
    
    neural_network = neural_networks_fourier.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS, n_nodes)
    #neural_network = neural_networks.neural_network(NUM_INPUTS, 1,HIDDEN_UNITS)





    #iny_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])

    
    value_int = neural_network.value(int_var)
    value_per1 = neural_network.value(per_var)
    value_per2 = neural_network.value(per_var1)
    value_per3 = neural_network.value(per_var2)
    value_per4 = neural_network.value(per_var3)
    weight = neural_network.weights
    w1 = list(weight.values())[0]
    w2 = list(weight.values())[1]
    w2 = tf.transpose(w2)
    regu_weight2 = .0001 * tf.math.reduce_sum(tf.norm(
                                                  w2,
                                                  ord= 2,
                                                  axis=None,
                                                  keepdims=None,
                                                  name=None
                                                  ))



    regu_weight1 = .0001 * tf.math.reduce_sum(tf.norm(
                                                      w1, #- entier_weight,
                                                      ord=2,
                                                      axis=None,
                                                      keepdims=None,
                                                      name=None
                                                      ))


    
    sol_int = tf.placeholder(tf.float64, [None, 1])
   

    
    
    
    loss_int = 1.*tf.square(value_int-sol_int)
    loss_per1 = 3. * tf.square(value_int-value_per1)
    loss_per2 = 3. *tf.square(value_int-value_per2)
    
    
    loss = tf.sqrt(tf.reduce_mean(loss_int + loss_per1 + loss_per2 ) + tf.square(regu_weight1) + tf.square(regu_weight2) )# + loss_bound))
    #loss = tf.sqrt(tf.reduce_mean(loss_int  ) + tf.square(regu_weight1) + tf.square(regu_weight2) )# + loss_bound))   #tf.sqrt(regu_weight1)
    
    
    train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':MAX_ITER})
    
    
    
    
    
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        
        sess.run(init)
        
        data = scipy.io.loadmat('data/input_data.mat')
        x = data['x'].flatten()[:,None]
        y_per1 = data['y'].flatten()[:,None]
        y_per2 = data['x1'].flatten()[:,None]
        
       
        int_draw = x
        
        
       
        
        
        f = problem.fun(int_draw)
        f = np.reshape(np.array(f), (BATCHSIZE, 1))
        
       
        
        
       
        
        
        print('start\n')
        train_scipy.minimize(sess, loss_callback=print_loss, fetches=[loss], feed_dict={sol_int:f,  int_var:x, per_var:y_per1, per_var1:y_per2})
        #train_scipy.minimize(sess, loss_callback=print_loss, fetches=[loss], feed_dict={sol_int:f,  int_var:x, per_var:y_per1, per_var1:y_per2, per_var2:y_per3, per_var3:y_per4})
        print('end\n')
        
        
        
        
        if DO_SAVE:
            save_name = 'test_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(MAX_ITER) + '_rs_' + str(SEED)+ '_nodes_' + str(n_nodes)
            save_path = saver.save(sess, save_name)
            print("Model saved in path: %s" % save_path)


        u_nn = neural_network.value(int_draw)
        ww = neural_network.weights
        bb = neural_network.biases
        w = sess.run(ww)
        b =sess.run(bb)
        #losses = sess.run(loss)
        print(sess.run(ww))
        print(sess.run(bb))
        
        np.savetxt("nn_fourier.txt", u_nn.eval(), fmt="%s")
# np.savetxt("loss_fourier.txt", loss.eval(), fmt="%s")







if __name__ == '__main__':
    main(sys.argv[1:])
    

