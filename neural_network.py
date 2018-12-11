from loadData import load_data

import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import keras 
from keras.models import Sequential
from keras.layers import Dense

from keras.models import Sequential
from keras.layers import Dense

def defineModel(num_vars,num_classes):
    model = Sequential()
    model.add(Dense(100, input_dim=num_vars, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
if __name__ == "__main__":

    # load data
    # used to train the model
    training_data, training_labels = load_data('mHealth_train.log')
    # used to select the best model
    #validation_data, validation_labels = load_data('mHealth_validation.log')
    # used to test the resulting model
    test_data, test_labels = load_data('mHealth_test.log')

    # variable to save the model
    save_model = False

    # define number of classes and variables in the data
    num_vars = len(training_data[0])
    num_classes = 13
    
    # create model
    model = defineModel(num_vars,num_classes)

    # fit model on training data
    model.fit(training_data,training_labels, epochs=10, batch_size=128)

    # test model on test data
    score = model.evaluate(test_data,test_labels,batch_size=128)
    print(score)
    # predictions are used to calculate precision and recall
    model_preds = model.predict_classes(test_data)
    print(model_preds)
    precision = precision_score(test_labels,model_preds,average='micro')
    recall = recall_score(test_labels,model_preds,average='micro')
    print("Precision on test set:",precision)
    print("Recall on test set:",recall)
    if save_model:
        model.save_weights('model_weights.h5')