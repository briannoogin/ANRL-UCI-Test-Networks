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
    validation_data, validation_labels = load_data('mHealth_validation.log')
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

    # train 25 models on the same training data and choose the model with the highest validation accuracy 
    max_acc = -1
    best_model = None
    num_iterations = 25
    for model_iteration in range(0,num_iterations):   
        # create model
        model = defineModel(num_vars,num_classes)
        # fit model on training data
        model.fit(training_data,training_labels, epochs=10, batch_size=128,verbose=0)
        # test model on validation data
        error,accuracy = model.evaluate(validation_data,validation_labels,batch_size=128,verbose=0)
        if(accuracy > max_acc):
            max_acc = accuracy
            best_model = model
        print("Trained Model", model_iteration + 1)
        print("Best Accuracy So Far ", max_acc)

    # predictions are used to calculate precision and recall
    val_model_preds = best_model.predict_classes(validation_data)
    test_model_preds = best_model.predict_classes(test_data)
    # report performance measures 
    val_precision = precision_score(validation_labels,val_model_preds,average='micro')
    val_recall = recall_score(validation_labels,val_model_preds,average='micro')
    test_precision = precision_score(test_labels,test_model_preds,average='micro')
    test_recall = precision_score(test_labels,test_model_preds,average='micro')
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