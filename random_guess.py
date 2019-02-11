
from collections import Counter
import random 
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
    print(cumulative_frequency)
    print(guess_preds[0:10])
def guess(cumulative_frequency):
    # set the seed for more deterministc outputs 
    random.seed(11)
    rand_num = random.random()
    for index in range(1,len(cumulative_frequency)):
        if rand_num <= cumulative_frequency[index] and rand_num >= cumulative_frequency[index-1]:
            return index
    return 0


        
