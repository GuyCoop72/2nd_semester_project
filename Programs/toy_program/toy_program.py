import tensorflow as tf
import numpy as np
import os
from scipy.sparse import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from file_loader import *
import cPickle
import time

INPUT_VECTOR_SIZE = 65536 # size of input vector
LAYER1_SIZE = 128 # size of hidden feature layer in generator and disc
G_NOISE_SIZE = 100 # size of input layer of noise to generator
mb_size = 64 # mini batch size
num_it = 100000 # set number of iteration of network training
output_folder = 'mb_64/'

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# discriminator setup
X0 = tf.placeholder(tf.bool, shape=[None, INPUT_VECTOR_SIZE])
X = tf.cast(X0, dtype=tf.float32)

D_W1 = tf.Variable(xavier_init([INPUT_VECTOR_SIZE, LAYER1_SIZE]))
D_b1 = tf.Variable(tf.zeros(shape=[LAYER1_SIZE]))

D_W2 = tf.Variable(xavier_init([LAYER1_SIZE, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

# generator setup
Z = tf.placeholder(tf.float32, shape=[None, G_NOISE_SIZE])

G_W1 = tf.Variable(xavier_init([G_NOISE_SIZE, LAYER1_SIZE]))
G_b1 = tf.Variable(tf.zeros(shape=[LAYER1_SIZE]))

G_W2 = tf.Variable(xavier_init([LAYER1_SIZE, INPUT_VECTOR_SIZE]))
G_b2 = tf.Variable(tf.zeros(shape=[INPUT_VECTOR_SIZE]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    # random noise sampling
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(sample):
    fig = plt.figure(figsize=(10,40))
    gs = gridspec.GridSpec(4,1)
    gs.update(wspace=0.05, hspace=0.05)

    sample = sample.reshape(4, 128, 128)
    for i, track in enumerate(sample):
        track = track.transpose()
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(track, cmap='Greys_r')

    return fig

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

Z_dim = G_NOISE_SIZE

# load the data in here
data = RockBars()

sess = tf.Session()
sess.run(tf.global_variables_initializer())


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

i = 0
# store losses
iter_array = []
D_loss_array = []
G_loss_array = []
loss_arrays = [iter_array, D_loss_array, G_loss_array]
start = time.clock()

# live plot losses
losses_fig = plt.figure()
G_loss_plt = losses_fig.add_subplot(211)
D_loss_plt = losses_fig.add_subplot(212)

for it in range(num_it):
    if it % 100 == 0:
        # store image of current sample
        sample = sess.run(G_sample, feed_dict={Z: sample_Z(1, Z_dim)})

        fig = plot(sample)
        plt.savefig((output_folder + 'image/{}.png').format(str(i).zfill(3)))
        plt.close(fig)

        # store vector of current sample
        samp_store = csc_matrix(sample)
        save_npz(output_folder + 'vector/' + str(i) + '.npz', samp_store)

        i += 1

    X_mb = data.get_minibatch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X0: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    # perform 2 runs of generator adjustment per iteration
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 10 == 0:
        # print losses
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print('time: {:.4}'.format(time.clock() - start))
        print()
        iter_array.append(it)
        D_loss_array.append(D_loss_curr)
        G_loss_array.append(G_loss_curr)

        # plot updating losses
        G_loss_plt.clear()
        G_loss_plt.plot(iter_array, G_loss_array)
        D_loss_plt.clear()
        D_loss_plt.plot(iter_array, D_loss_array)
        plt.draw()
        plt.pause(0.0001)




# store the losses
f = open(output_folder + "losses.data", 'wb')
cPickle.dump(loss_arrays, f)
f.close()