from loadData import load_data
import tensorflow as tf
import numpy as np
import random
import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import csv
import keras 
from keras.models import Sequential
from keras.layers import Dense

from keras.models import Sequential
from keras.layers import Dense

def defineModel(num_vars,num_classes):
    model = Sequential()
    model.add(Dense(100, input_dim=num_vars, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
if __name__ == "__main__":
    # load data
    training_data, training_labels = load_data('mHealth_train.log')
    test_data, test_labels = load_data('mHealth_test.log')

    # define number of classes and variables in the data
    num_vars = len(training_data[0])
    num_classes = 13
    
    # create model
    model = defineModel(num_vars,num_classes)

    # fit model on training data
    model.fit(training_data,training_labels, epochs=10, batch_size=32)

    # test model on test data
    score = model.evaluate(test_data,test_labels,batch_size=128)
    print(score)