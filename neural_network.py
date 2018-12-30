# set the RNG seeds
import numpy as np
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(2)
from loadData import load_data


import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

import keras 
from keras.models import Sequential
from keras.layers import Dense,Input,add, Lambda
from keras import regularizers
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.models import Model
from keras.backend import ones
from keras.backend import zeros

from ann_visualizer.visualize import ann_viz;

import time
# assumes that the hidden units and the regularization constant are consistent throughout the network
# returns the baseline_model 
def define_baseline_model(num_vars,num_classes,hidden_units,regularization):
    model = Sequential()
    # one input layer
    # 10 hidden layers
    model.add(Dense(units=hidden_units, input_dim=num_vars, activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    # one output layer
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_baseline_functional_model(num_vars,num_classes,hidden_units, regularization):
    # one input layer
    input_layer = Input(shape = (num_vars,))
    # 10 hidden layers
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(input_layer)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    hidden_layers = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(hidden_layers)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax')(hidden_layers)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# lambda function to add physical nodes in the network 
# input: tensors from other layers 
def add_node_layers(input_tensors):
    return input_tensors
# returns fixed guard model with 10 hidden layers
# f1 = fog node 2 = 1st hidden layer
# f2 = fog node 2 = 2nd and 3rd hidden layer
# f3 = fog node 3 = 4th-6th hidden layers
# c = cloud node = 7th-10th hidden layer and output layer 
def define_fixedguard_model(num_vars,num_classes,hidden_units,regularization,failures_rates):
    # one input layer
    input_layer = Input(shape = (num_vars,))
    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(input_layer)
    #connection_f1f2 = multiply([connection_f1f2,)
    lambda_layer = Lambda(add_node_layers)
    connection_f1f2 = lambda_layer([f1])
    # second fog node
    f2 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(connection_f1f2)
    f2 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(f2)
    connection_f2f3 = lambda_layer([f2])
    # third fog node
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(connection_f2f3)
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(f3)
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(f3)
    connection_f3c = lambda_layer([f3])
    # cloud node
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(connection_f3c)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(cloud)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(cloud)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax')(cloud)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# returns active guard model
def define_activeguard_model(num_vars,num_classes,hidden_units,regularization):
    # one input layer
    input_layer = Input(shape = (num_vars,))
    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(input_layer)
    connection_f1f2 = zeros(f1.shape)
    #connection_f1f2 = multiply([connection_f1f2,)
    connection_f1f2 = add([f1,connection_f1f2])
    # second fog node
    f2 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(connection_f1f2)
    f2 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(f2)
    connection_f2f3 = zeros(f2.shape)
    connection_f2f3 = add([f2,connection_f2f3])
    # third fog node
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(connection_f2f3)
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(f3)
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(f3)
    connection_f3f4 = zeros(f3.shape)
    connection_f3f4 = add([f3,connection_f3f4])
    # cloud node
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(connection_f3f4)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(cloud)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(cloud)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization))(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax')(cloud)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# save weights of network 
def save_weights(model):
    weights_name = model.layers() + 'model_weights.h5'
    model.save_weights(weights_name)

# load model from the weights 
# model type 0 = baseline model
# model type 1 = fixed guard model
# model type 2 = active guard model
def load_model(input_size, output_size, hidden_units, regularization, weights_path,model_type):
    if model_type == 0:
        model = define_baseline_functional_model(input_size,output_size,hidden_units,regularization)
    if model_type == 1:
        model = define_fixedguard_model(input_size,output_size,hidden_units,regularization,[.99,.96,.92])
    else:
        model = define_activeguard_model(input_size,output_size,hidden_units,regularization)
    model.load_weights(weights_path)
    #print_weights(model)
    return model

# change the weights of a layer of a model
def change_weights(model,layer_index):
    weights = model.get_weights()
    layer_dim= weights[layer_index].shape
    # make numpy of zeros for the layer weight 
    layer = np.zeros(layer_dim)
    weights[layer_index] = layer
    model.set_weights(weights)
    return model

def print_weights(model):
    for layer in model.layers: 
        print(layer.get_config(),layer.get_weights())

# trains and returns the model 
def train_model(training_data,training_labels,validation_data,validation_labels):
    # variable to save the model
    save_model = False

    # train 5 models on the same training data and choose the model with the highest validation accuracy 
    max_acc = -1
    best_model = None
    num_iterations = 1
    for model_iteration in range(0,num_iterations):   
        # create model
        model = define_baseline_functional_model(num_vars,num_classes,50,.01)
        # fit model on training data
        model.fit(training_data,training_labels, epochs=100, batch_size=128,verbose=0,shuffle = True)
        # test model on validation data
        error,accuracy = model.evaluate(validation_data,validation_labels,batch_size=128,verbose=0)
        if(accuracy > max_acc):
            max_acc = accuracy
            best_model = model
        print("Trained Model", model_iteration + 1)
        print("Best Accuracy So Far ", max_acc)

    # predictions are used to calculate precision and recall
    val_model_preds = predict_classes(best_model,validation_data)
    # report performance measures 
    val_precision = precision_score(validation_labels,val_model_preds,average='macro')
    val_recall = recall_score(validation_labels,val_model_preds,average='macro')
   
    # print results
    print()
    print("Accuracy on validation set:", max_acc)
    print("Precision on validation set:",val_precision)
    print("Recall on validation set:",val_recall, '\n')
    if save_model:
        best_model.save_weights('normalized data 10 layers 100 units .01 reg adam corrected_training model_weights.h5')
    return model

# returns the test performance measures 
def test_model(model,test_data,test_labels):
    test_model_preds = predict_classes(model,test_data)
    test_precision = precision_score(test_labels,test_model_preds,average='macro')
    test_recall = precision_score(test_labels,test_model_preds,average='macro')
    test_error,test_accuracy = model.evaluate(test_data,test_labels,batch_size=128,verbose=0)
    print("Accuracy on test set:", test_accuracy)
    print("Precision on test set:",test_precision)
    print("Recall on test set:",test_recall)

# returns the classes prediction from a Keras functional model
def predict_classes(functional_model,data):
    y_prob = functional_model.predict(data) 
    return y_prob.argmax(axis=-1)

if __name__ == "__main__":
    # load data
    training_data, training_labels = load_data('mHealth_train.log')
    validation_data, validation_labels = load_data('mHealth_validation.log')
    test_data, test_labels = load_data('mHealth_test.log')

    # define number of classes and variables in the data
    num_vars = len(training_data[0])
    num_classes = 13

    load_weights = True
    if load_weights:
        path = '10 layers 50 units .01 reg adam corrected_training model_weights.h5'
        model = load_model(input_size = num_vars, output_size = num_classes, hidden_units = 50, regularization = .01, weights_path = path, model_type = 1)
        #plot_model(model,to_file = "model.png",show_shapes = True)
        #ann_viz(model, title="Artificial Neural network - Model Visualization")
    else:
        start = time.time()
        model = train_model(training_data,training_labels,validation_data,validation_labels)
        end = time.time()
        print("Time elapsed:", end-start)
    test_model(model,validation_data,validation_labels)
    test_model(model,test_data,test_labels)
  