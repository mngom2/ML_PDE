import tensorflow as tf
import numpy as np 
import obj_fun
import poisson_problem
import neural_networks
import neural_networks_fourier
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker
from numpy import linalg as LA

rc('font', **{'size':12, 'family':'serif', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)


def draw_magnitude_int_loss(x_range, y_range, session, x_qual, y_qual):
	x = np.reshape(np.linspace(x_range[0], x_range[1], x_qual), (x_qual,1))
	f = problem.fun(x)

	loss_int_magnitude = np.sqrt(session.run(loss_int, feed_dict={int_var: x, sol_int:f}))

	return np.reshape(loss_int_magnitude, (x_qual, 1))


def draw_magnitude_of_err_2d(x_range, y_range, exact_sol, x_qual, y_qual, neural_net_sol):
	x = np.reshape(np.linspace(x_range[0], x_range[1], x_qual), (x_qual,1))
	mesh = x

	u_sol = exact_sol(mesh)

	neural_net_sol_mesh = neural_net_sol(mesh.astype(np.float64)).eval()


	err_vec = np.zeros(x_qual)

	for i in range(x_qual):
		err_vec[i] = np.sqrt((u_sol[i]-neural_net_sol_mesh[i])**2)

	err_vec = np.reshape(err_vec, (x_qual,1))
	return err_vec, neural_net_sol_mesh, mesh

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

NUM_STEPS = 1
NUM_INPUTS = 1
BATCHSIZE = 100*100
#HIDDEN_UNITS = [n_nodes] #, n_nodes, n_nodes]
myfile = open('errmode_5.txt', 'w')
modes = 5



for n_nodes in range(1,16):
    HIDDEN_UNITS = [n_nodes]
    restore_name = 'test_model/1_layer_sq_loss_100_m_iter_50000_rs_42_nodes_'+str(n_nodes)+'_modes_'+str(modes)
    #problem = poisson_problem.poisson_1d()
    problem = obj_fun.objfun()

    neural_network = neural_networks_fourier.neural_network(NUM_INPUTS, 1, HIDDEN_UNITS,n_nodes)
    #neural_network = neural_networks.neural_network(NUM_INPUTS, 1, HIDDEN_UNITS)
    biases = neural_network.biases

    int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])
    bou_var = tf.placeholder(tf.float64, [None, NUM_INPUTS])

    value_int = neural_network.value(int_var)
    value_bou = neural_network.value(bou_var)

    #grad, grad_grad = neural_network.derivatives(int_var)
    #= neural_network.second_derivatives(int_var)

    sol_int = tf.placeholder(tf.float64, [None, 1])
    sol_bou = tf.placeholder(tf.float64, [None, 1])

    loss_int = 0 #tf.square(grad_grad[0]+sol_int)
    loss_bou = tf.square(value_bou-sol_bou)
    loss = tf.reduce_mean(loss_int + loss_bou)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, restore_name)
        print(sess.run(biases))

        err, nn,xx = draw_magnitude_of_err_2d(problem.range, problem.range, problem.fun, 41, 1, neural_network.value)
        myfile.write("%s\n" % LA.norm(err))
        
        


        np.savetxt("nn.txt", nn, fmt="%s")

myfile.close()

				
		
