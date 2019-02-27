
from collections import Counter
import random 
from keras.models import Model
import keras.backend as K
from sklearn.metrics import accuracy_score
import numpy as np

# use a model with trained weights to guess if there are no connections 
def model_guess(model,train_labels,test_data,test_labels):
    preds = model.predict(test_data)
    preds = np.argmax(preds,axis=1)
    # check if the connection is 0 which means that there is no data flowing in the network
    f3 = model.get_layer(name = "F1F2_F3").output
    # get the output from the layer
    output_model = Model(inputs = model.input,outputs=f3)
    f3_output = output_model.predict(test_data)
    no_connection_flow = np.array_equal(f3_output,f3_output * 0)
    # there is no connection flow, make random guess 
    if no_connection_flow:
        print("There is no data flow in the network")
        preds = random_guess(train_labels,test_data)
    acc = accuracy_score(test_labels,preds)
    print(acc)
    K.clear_session()
    return acc

# function returns a array of predictions based on random guessing
# random guessing is determined by the class distribution from the training data. 
# input: list of training labels
# input: matrix of test_data, rows are examples and columns are variables 
def random_guess(train_labels,test_data):
    # count the frequency of each class
    class_frequency = Counter(train_labels)
    # sort by keys and get the values
    sorted_class_frequency = list(dict(sorted(class_frequency.items())).values())
    total_frequency = len(train_labels)
    # find relative frequency 
    sorted_class_frequency = [freq / total_frequency for freq in sorted_class_frequency]
    # append a 0 to the beginning of a new list
    cumulative_frequency = [0] + sorted_class_frequency
    # calculate cumulative relative frequency 
    for index in range(1,len(cumulative_frequency)):
        cumulative_frequency[index] += cumulative_frequency[index-1]
    # make a guess for each test example
    guess_preds = [guess(cumulative_frequency) for example in test_data]
    return guess_preds
# makes a random number and determines a class based on the cumulative frequency
def guess(cumulative_frequency):
    # set the seed for more deterministc outputs 
    random.seed(11)
    rand_num = random.random()
    for index in range(1,len(cumulative_frequency)):
        if rand_num <= cumulative_frequency[index] and rand_num >= cumulative_frequency[index-1]:
            return index
    return 0


        