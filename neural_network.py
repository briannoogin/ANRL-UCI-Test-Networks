from loadData import load_data
import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import csv

def addLayer(x, input_size, output_size, layer_num, softmax=True):
    var_name = 'W_' + str(layer_num)
    W = tf.Variable(tf.truncated_normal([input_size, output_size]), name=var_name)

    bias_name = 'B_' + str(layer_num)
    b = tf.Variable(tf.truncated_normal([output_size]), name=bias_name)

    weights[var_name] = W

    if softmax:
        return tf.nn.softmax(tf.matmul(x, W) + b)
    else:
        return tf.nn.relu(tf.matmul(x, W) + b)

def make_network(input_data, arch):
    # first layer
    nn = addLayer(x=input_data, 
                    input_size=num_vars, 
                    output_size=arch[0], 
                    layer_num=1,
                    softmax = False)

    # hidden layers
    for layer in range(1, len(arch)):
        nn = addLayer(x=nn, 
                        input_size=arch[layer-1], 
                        output_size=arch[layer], 
                        layer_num=layer + 1,
                        softmax = False)

    # output layer
    nn = addLayer(x=nn, 
                    input_size=arch[len(arch) - 1], 
                    output_size=12, 
                    layer_num='out', 
                    softmax=True)

    return nn

if __name__ == "__main__":
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
    weights = {}
    normalize_data = True
    save_weights = False
    # number of variables we consider in each input
    num_vars = 23
    num_classes = 12
    iter_ = 1000
    lr = 1e-1
    batch_size = 32

    # input placeholder
    X = tf.placeholder(tf.float32, [None, num_vars])

    # y_true
    Y_ = tf.placeholder(tf.float32, [None, num_classes])

    # entire model
    network_prediction = make_network(X, icnn_arch)

    #cost = tf.reduce_mean(tf.square(network_prediction - Y_))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_prediction, labels=Y_) )
    train = tf.train.AdamOptimizer(lr).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print('Data snippit...')
        # load same data as cvx fit
        x_train, y_train = load_data('mHealth_train.log')
        x_test, y_test = load_data('mHealth_test.log')
        
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
        pred = sess.run(network_prediction, feed_dict={X: x_train})
        error = []

        #print('{0:25} {1}'.format('pred', 'real'))
        for j in range(len(pred)):
            #print('{0:<25} {1}'.format(str(pred[j][0]), y_test[j]))
            error.append((y_train[j] - pred[j][0]) ** 2)

        print('\nTotal train size: ', len(y_train))
        print('Train mse: ', sum(error) / len(error))
        print("prediction",pred)
        print("labels",y_train)
        print("acc",accuracy_score(y_train,pred))
    # ======================================================================== #
        # test on validation set
        print('\nTesting on validation-set...')
        pred = sess.run(network_prediction, feed_dict={X: x_test})
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
        if save_weights:
            # save weights
            tvars = tf.trainable_variables()
            tvars_vals = sess.run(tvars)
            # make weights into a dictionary
            tvars_dict = {}
            for index in range(len(tvars_vals)):
                tvars_dict[tvars[index]] = tvars_vals[index]
            # write to a csv file
            with open('weight.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in tvars_dict.items():
                    writer.writerow([key.name, value])
    
