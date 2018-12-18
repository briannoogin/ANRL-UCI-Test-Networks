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
from keras.models import Sequential
from keras.layers import Dense

def defineModel(num_vars,num_classes):
    model = Sequential()
    model.add(Dense(units=100, input_dim=num_vars, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=100,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Dense(units=100,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=num_classes, activation='softmax'))
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

    # standardize data based on Z-score
    #scaler = StandardScaler()
    #scaler.fit(training_data)
    #training_data = scaler.transform(training_data)
    # validation_data = scaler.transform(validadtion_data)
    #test_data = scaler.transform(test_data)

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
    error,accuracy = model.evaluate(test_data,test_labels,batch_size=128)
    # predictions are used to calculate precision and recall
    model_preds = model.predict_classes(test_data)

    # report performance measures 
    precision = precision_score(test_labels,model_preds,average='micro')
    recall = recall_score(test_labels,model_preds,average='micro')
    print("Accuracy on test set:", accuracy)
    print("Precision on test set:",precision)
    print("Recall on test set:",recall)
    if save_model:
        model.save_weights('model_weights.h5')