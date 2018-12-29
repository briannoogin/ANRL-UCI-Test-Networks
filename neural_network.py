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
from keras.layers import Dense, Input
from keras import regularizers
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.models import Model

from ann_visualizer.visualize import ann_viz;

import time
# assumes that the hidden units and the regularization constant are consistent throughout the network
# returns the baseline_model 
def define_baseline_model(num_vars,num_classes,hidden_units,regularization):
    model = Sequential()
    # one input layer
    # 10 hidden layers
    model.add(Dense(units=hidden_units, input_dim=num_vars, activation='relu',kernel_regularizer=regularizers.l2(regularization),name ="1st IOT Block"))
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

# returns fixed guard model
def define_fixedguard_model(num_vars,num_classes,hidden_units,regularization):
    print('hi')

# returns active guard model
def define_activeguard_model(num_vars,num_classes,hidden_units,regularization):
    print('hi2')

# save weights of network 
def save_weights(model):
    weights_name = model.layers() + 'model_weights.h5'
    model.save_weights(weights_name)

# load model from the weights 
def load_model(input_size, output_size, hidden_units, regularization, weights_path):
    model = define_baseline_functional_model(input_size,output_size,hidden_units,regularization)
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
        model = define_baseline_model(num_vars,num_classes,100,.01)
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
        best_model.save_weights('10 layers 100 units .01 reg adam corrected_training model_weights.h5')
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

def print_weights(model):
    for layer in model.layers: 
        print(layer.get_config(), len(layer.get_weights()[1]))
# returns the classes prediction from a Keras functional model
def predict_classes(functional_model,data):
    y_prob = model.predict(data) 
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
        model = load_model(input_size = num_vars, output_size = num_classes, hidden_units = 50, regularization = .01, weights_path = path)
        #plot_model(model,to_file = "model.png",show_shapes = True)
        #ann_viz(model, title="Artificial Neural network - Model Visualization")
    else:
        start = time.time()
        model = train_model(training_data,training_labels,validation_data,validation_labels)
        end = time.time()
        print("Time elapsed:", end-start)
    test_model(model,validation_data,validation_labels)
    test_model(model,test_data,test_labels)
  