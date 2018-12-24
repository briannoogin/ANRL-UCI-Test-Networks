# set the RNG seeds
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from loadData import load_data
import numpy as np


import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers


# assumes that the hidden units and the regularization constant are consistent throughout the network
def defineModel(num_vars,num_classes,hidden_units,regularization):
    model = Sequential()
    model.add(Dense(units=hidden_units, input_dim=num_vars, activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l2(regularization)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# save weights of network 
def save_weights(model):
    weights_name = model.layers() + 'model_weights.h5'
    model.save_weights(weights_name)
# load model from the weights 
def load_model(input_size, output_size, hidden_units, regularization, weights_path):
    model = defineModel(input_size,output_size,hidden_units,regularization)
    model.load_weights(weights_path)
    weights = model.get_weights()
    change_weights(model,weights,1)
    return model
# change the weights of a layer of a model
def change_weights(model,weights,layer_index):
    layer_dim= weights[layer_index].shape
    # make numpy of zeros for the layer weight 
    layer = np.zeros(layer_dim)
    weights[layer_index] = layer
    model.set_weights(weights)
    print(weights)
# trains and returns the model 
def train_model(training_data,training_labels,validation_data,validation_labels,test_data,test_labels):
    # variable to save the model
    save_model = True

    # train 10 models on the same training data and choose the model with the highest validation accuracy 
    max_acc = -1
    best_model = None
    num_iterations = 5
    for model_iteration in range(0,num_iterations):   
        # create model
        model = defineModel(num_vars,num_classes,150,.25)
        # fit model on training data
        model.fit(training_data,training_labels, epochs=100, batch_size=32,verbose=0)
        # test model on validation data
        error,accuracy = model.evaluate(validation_data,validation_labels,batch_size=32,verbose=0)
        if(accuracy > max_acc):
            max_acc = accuracy
            best_model = model
        print("Trained Model", model_iteration + 1)
        print("Best Accuracy So Far ", max_acc)

    # predictions are used to calculate precision and recall
    val_model_preds = best_model.predict_classes(validation_data)
    test_model_preds = best_model.predict_classes(test_data)
    # report performance measures 
    val_precision = precision_score(validation_labels,val_model_preds,average='macro')
    val_recall = recall_score(validation_labels,val_model_preds,average='macro')
    test_precision = precision_score(test_labels,test_model_preds,average='macro')
    test_recall = precision_score(test_labels,test_model_preds,average='macro')
    test_error,test_accuracy = best_model.evaluate(test_data,test_labels,batch_size=128,verbose=0)
    # print results
    print()
    print("Accuracy on validation set:", max_acc)
    print("Precision on validaion set:",val_precision)
    print("Recall on validation set:",val_recall, '\n')

    print("Accuracy on test set:", test_accuracy)
    print("Precision on test set:",test_precision)
    print("Recall on test set:",test_recall)
    if save_model:
        best_model.save_weights('model_weights.h5')
    return model
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
        path = '150 .25 reg 100 epochs model_weights.h5'
        model = load_model(input_size = num_vars, output_size = num_classes, hidden_units = 150, regularization = .25, weights_path = path)
    else:
        model = train_model(training_data,training_labels,validation_data,validation_labels,test_data,test_labels)
  