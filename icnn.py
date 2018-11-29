from load import load_data
import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import preprocessing

# fixing seed for reproduction of results
np_rand = random.randint(0,10000)
from numpy.random import seed
seed(np_rand)

tf_rand = random.randint(0,10000)
from tensorflow import set_random_seed
set_random_seed(tf_rand)

print('np seed: ', np_rand)
print('tf seed: ', tf_rand)

# for abstracting model creation
#icnn_arch = [256, 128, 128, 64, 32, 32]
icnn_arch = [2000, 2000]
print_weights = False
clipped_weights = {}
normalize_data = False

# number of variables we consider in each input
num_vars = 14

iter_ = 90000
lr = 1e-1
batch_size = 32

init = tf.initialize_all_variables()

def clipped_fc(x, input_size, output_size, layer_num, softmax=True):
    var_name = 'W_' + str(layer_num)
    W = tf.Variable(tf.truncated_normal([input_size, output_size]), name=var_name)

    bias_name = 'B_' + str(layer_num)
    b = tf.Variable(tf.truncated_normal([output_size]), name=bias_name)

    clipped_weights[var_name] = W

    if softmax:
        return tf.nn.relu(tf.matmul(x, W) + b)
    else:
        return tf.matmul(x, W) + b

def fcnn(input_data, arch):
    # first layer
    nn = clipped_fc(x=input_data, 
                    input_size=num_vars, 
                    output_size=arch[0], 
                    layer_num=1)

    # hidden layers
    for layer in range(1, len(arch)):
        nn = clipped_fc(x=nn, 
                        input_size=arch[layer-1], 
                        output_size=arch[layer], 
                        layer_num=layer + 1)

    # output layer, don't use softmax
    nn = clipped_fc(x=nn, 
                    input_size=arch[len(arch) - 1], 
                    output_size=1, 
                    layer_num='out', 
                    softmax=False)

    return nn


# input placeholder
X = tf.placeholder(tf.float32, [None, num_vars])

# y_true
Y_ = tf.placeholder(tf.float32, [None, 1])

# entire model
icnn_out = fcnn(X, icnn_arch)

# create clipping ops for each weight variable in our icnn
clip_ops = [tf.assign(clipped_weights[w], tf.maximum(0., clipped_weights[w])) for w in clipped_weights]

cost = tf.reduce_mean(tf.square(icnn_out - Y_))
train = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    print('Data snippit...')
    # load same data as cvx fit
    x_train, y_train = load_data('air_quality_train', omit=True)
    x_test, y_test = load_data('air_quality_test', omit=True)
    
    print(x_train[0])
    print(y_train[0])

    # convert y_train to tf usable shape (not (148,) np shape)
    y_train = np.reshape(y_train, [len(y_train),1])

    # normalize input data
    if normalize_data:
        x_train = sklearn.preprocessing.scale(x_train)
        x_test = sklearn.preprocessing.scale(x_test)

    for i in range(iter_):
        # sklearn shuffle to get minibatch
        shuffled_x, shuffled_y = sklearn.utils.shuffle(x_train, y_train, n_samples=batch_size)

        # train
        _, current_cost = sess.run([train, cost], feed_dict={X: shuffled_x, Y_: shuffled_y})

        # apply weight clipping
        for op in clip_ops:
            sess.run(op)

        # print training progress
        if i % 100 == 0:
            print('Iter: {0} Train mse: {1}'.format(i, current_cost))
            
            # print out weights to make sure > 0 
            if print_weights:
                tvars = tf.trainable_variables()
                tvars_vals = sess.run(tvars)
                for var, val in zip(tvars, tvars_vals):
                    print(var.name, val)

# ======================================================================= #
    # print MSE on entire training set
    print('\nTesting on train-set...')
    pred = sess.run(icnn_out, feed_dict={X: x_train})
    error = []

    #print('{0:25} {1}'.format('pred', 'real'))
    for j in range(len(pred)):
        #print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
        error.append((y_train[j] - pred[j][0]) ** 2)

    print('\nTotal train size: ', len(y_train))
    print('Train mse: ', sum(error) / len(error))

# ======================================================================== #
    # test on validation set
    print('\nTesting on validation-set...')
    pred = sess.run(icnn_out, feed_dict={X: x_test})
    error = []

    #print('{0:25} {1}'.format('pred', 'real'))
    for j in range(len(pred)):
        #print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
        error.append((y_test[j] - pred[j][0]) ** 2)

    print('\nTotal test size: ', len(y_train))
    print('Test mse: ', sum(error) / len(error))

    print('\nNormalized Data:', normalize_data)
    print('Architecture:',icnn_arch)
    print('lr_:',lr)
    print('iterations:',iter_)

'''
## legacy icnn construction method

# fc1
W_1 = tf.Variable(tf.truncated_normal([num_vars, fc1_size]))
b_1 = tf.Variable(tf.truncated_normal([fc1_size]))
fc1 = tf.nn.softmax(tf.matmul(X, W_1) + b_1)

# fc2 
W_2 = tf.Variable(tf.truncated_normal([fc1_size,fc2_size]))
b_2 = tf.Variable(tf.truncated_normal([fc2_size]))
fc2 = tf.nn.softmax(tf.matmul(fc1, W_2) + b_2)

# fc3
W_3 = tf.Variable(tf.truncated_normal([fc2_size,1]))
b_3 = tf.Variable(tf.truncated_normal([1]))
out = tf.matmul(fc2, W_3) + b_3

# clipping op so weights stay in [0, inf)
clip_W_1 = W_1.assign(tf.maximum(0., W_1))
clip_W_2 = W_2.assign(tf.maximum(0., W_2))
clip_W_3 = W_3.assign(tf.maximum(0., W_3))
clip = tf.group(clip_W_1, clip_W_2, clip_W_3)
'''
