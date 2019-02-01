# set the RNG seeds
import numpy as np
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(2)
from loadData import load_data


import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import keras 
from keras.utils import plot_model
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import datetime
import os

from ActiveGuard import define_active_guard_model_with_connections
from FixedGuard import define_model_with_connections, define_model_with_nofogbatchnorm_connections
from Baseline import define_baseline_functional_model

# fails node by making the physical node return 0
# node_array: bit array, 1 corresponds to alive, 0 corresponds to failure
def fail_node(model,node_array):
    for index,node in enumerate(node_array):
        # node failed
        if node == 0:
            layer_name = "fog" + str(index + 1) + "_output_layer"
            layer = model.get_layer(name=layer_name)
            layer_weights = layer.get_weights()
            #print(layer_weights)
            # make new weights for the connections
            new_weights = np.zeros(layer_weights[0].shape)
            #new_weights[:] = np.nan # set weights to nan
            # make new weights for biases
            new_bias_weights = np.zeros(layer_weights[1].shape)
            #new_bias_weights[:] = np.nan # set weights to nan
            layer.set_weights([new_weights,new_bias_weights])
            print(layer_name, "was failed")

# trains and returns the model 
def train_model(training_data,training_labels,model_type):
    # variable to save the model
    save_model = True

    # train 1 model on the same training data and choose the model with the highest validation accuracy 
    num_iterations = 1
    for model_iteration in range(0,num_iterations):   
        # create model
        if model_type == 0:
            model = define_baseline_functional_model(num_vars,num_classes,50,0)
        elif model_type == 1:
            model = define_model_with_connections(num_vars,num_classes,50,0,[.99,.96,.92])
        elif model_type == 2:
            model = define_active_guard_model_with_connections(num_vars,num_classes,10,0,[.99,.96,.92])
        elif model_type == 3:
            # survive_rates = [.70,.75,.80]
            # failure_rates = [.3,.25,.20]
            survive_rates = [.70,.75,.85]
            model = define_model_with_nofogbatchnorm_connections(num_vars,num_classes,50,0,survive_rates)
        else:
            raise ValueError("Incorrect model type")
        # fit model on training data
        model.fit(training_data,training_labels, epochs=25, batch_size=128,verbose=1,shuffle = True)

    now = datetime.datetime.now()
    date = str(now.month) + '-' + str(now.day) + '-' + str(now.year)
    path = 'weights/' + date
    if not os.path.exists(path):
        os.mkdir(path)
    if save_model:
        model.save_weights(path + '/50 units 10 layers with normal connections [.70,.75,.85] weights he_normal imputed 0s no batch_norm .25 dropout adam batchnormcloud 25 epochs' + '.h5')
    return model

# load model from the weights 
# model type 0 = baseline model
# model type 1 = fixed guard model
# model type 2 = active guard model 
def load_model(input_size, output_size, hidden_units, regularization, weights_path,model_type):
    if model_type == 0:
        model = define_baseline_functional_model(input_size,output_size,hidden_units,regularization)
    elif model_type == 1:
        model = define_model_with_connections(input_size,output_size,hidden_units,regularization,[.99,.96,.92])
    elif model_type == 2:
        model = define_active_guard_model_with_connections(input_size,output_size,hidden_units,regularization,[.99,.96,.92])
    elif model_type == 3:
        failure_rates = [.70,.75,.85]
        model = define_model_with_nofogbatchnorm_connections(num_vars,num_classes,hidden_units,0,failure_rates)
    else:
        raise ValueError("Incorrect model type")
    model.load_weights(weights_path,by_name=True)
    #print_weights(model)
    return model

# returns the test performance measures 
def test_model(model,test_data,test_labels,set_name):
    test_model_preds = predict_classes(model,test_data)
    test_precision = precision_score(test_labels,test_model_preds,average='micro')
    test_recall = precision_score(test_labels,test_model_preds,average='micro')
    test_error,test_accuracy = model.evaluate(test_data,test_labels,batch_size=128,verbose=0)
    if set_name == 'validation':
        print("Accuracy on validation set:", test_accuracy)
        print("Precision on validation set:",test_precision)
        print("Recall on validation set:",test_recall)
    if set_name == 'test':
        print("Accuracy on test set:", test_accuracy)
        print("Precision on test set:",test_precision)
        print("Recall on test set:",test_recall)

# returns the classes prediction from a Keras functional model
def predict_classes(functional_model,data):
    y_prob = functional_model.predict(data) 
    return y_prob.argmax(axis=-1)

# prints the output of a layer of the model
def print_layer_output(model,data,layer_name):
        layer_output = model.get_layer(name = layer_name).output
        output_model = Model(inputs = model.input,outputs=layer_output)
        intermediate_output = output_model.predict(data)
        #print(intermediate_output.shape)
        print(intermediate_output)
        # calculates the number of zeros in the output
        # used for checking the effects of dropping out nodes
        non_zeros = np.count_nonzero(intermediate_output)
        zeros = intermediate_output.size - non_zeros
        print("Number of zeros in the output:",zeros)

def evaluate_withFailures(model,test_data,test_labels):
    failure_list = [
        [1,1,1], # no failures
        [0,1,1], # fog node 1 fails
        [1,0,1], # fog node 2 fails
        [1,1,0], # fog node 3 fails
    ]
    # keep track of original weights so it does not get overwritten by fail_node
    original_weights = model.get_weights()
    for failure in failure_list:
        print(failure)
        fail_node(model,failure)
        test_model(model,test_data,test_labels,'test')
        # reset the model with the original weights 
        model.set_weights(original_weights)
        
if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # load data
    training_data, training_labels = load_data('mHealth_train.log')
    test_data, test_labels = load_data('mHealth_test.log')

    # define number of classes and variables in the data
    num_vars = len(training_data[0])
    num_classes = 13

    # define model type
    model_type = 3

    load_weights = False
    if load_weights:
        path = 'weights/1-31-2019/50 units 10 layers with regular connections [.70,.75,.85] weights he_normal imputed 0s no batch_norm.h5'
        model = load_model(input_size = num_vars, output_size = num_classes, hidden_units = 50, regularization = 0, weights_path = path, model_type = model_type)
        #plot_model(model,to_file = "model_with_ConnectionsAndBatchNorm.png",show_shapes = True)
    else:
        model = train_model(training_data,training_labels,model_type=model_type)
    evaluate_withFailures(model,test_data,test_labels)

    # check if output layer has all zeros
    # print('F1F2_F3 Output Before Failure')
    # print_layer_output(model,test_data,'F1F2_F3')
    # fail_node(model,[1,0,1])
    # print('F1F2_F3 Output After Failure')
    # print_layer_output(model,test_data,'F1F2_F3')
  