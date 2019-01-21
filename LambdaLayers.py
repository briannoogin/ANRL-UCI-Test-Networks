from keras.layers import add
import keras

# lambda function to add physical nodes in the network 
# input: list of tensors from other layers 
# input_tensors[0] = output layer of first node
# input_tensors[1] = output layer of second node
# returns the sum of the output layers
def add_node_layers(input_tensors):
    first_input = input_tensors[0]
    second_input = input_tensors[1]
    # TODO: check for NaNs, used for failure cases
    return add([first_input,second_input])

# lambda function to add the beginning physical node in the network
# input: one tensor
def add_first_node_layers(input_tensor):
    first_input = input_tensor
    return first_input