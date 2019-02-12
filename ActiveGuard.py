from keras.models import Sequential
from keras.layers import Dense,Input,add,multiply,Lambda, BatchNormalization, Activation
import keras.backend as K
import tensorflow as tf
from LambdaLayers import add_first_node_layers
from LambdaLayers import add_node_layers
from keras import regularizers
from keras import optimizers
from keras.models import Model
from keras.backend import constant
import numpy as np

# returns active guard model with 10 hidden layers
# f1 = fog node 2 = 1st hidden layer
# f2 = fog node 2 = 2nd and 3rd hidden layer
# f3 = fog node 3 = 4th-6th hidden layers
# c = cloud node = 7th-10th hidden layer and output layer 
def define_active_guard_model_with_connections(num_vars,num_classes,hidden_units,regularization,survive_rates):
    # calculate connection weights
    # naming convention:
    # ex: f1f2 = connection between fog node 1 and fog node 2
    # ex: f2c = connection between fog node 2 and cloud node
    connection_weight_f1f2 = 1
    connection_weight_f1f3 = survive_rates[0] / (survive_rates[0] + survive_rates[1])
    connection_weight_f2f3 = survive_rates[1] / (survive_rates[0] + survive_rates[1])
    connection_weight_f2c = survive_rates[1] / (survive_rates[1] + survive_rates[2])
    connection_weight_f3c = survive_rates[2] / (survive_rates[1] + survive_rates[2])

    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_f1f2 = Lambda((lambda x: x * connection_weight_f1f2), name = "connection_weight_f1f2")
    multiply_weight_layer_f1f3 = Lambda((lambda x: x * connection_weight_f1f3), name = "connection_weight_f1f3")
    multiply_weight_layer_f2f3 = Lambda((lambda x: x * connection_weight_f2f3), name = "connection_weight_f2f3")
    multiply_weight_layer_f2c = Lambda((lambda x: x * connection_weight_f2c), name = "connection_weight_f2c")
    multiply_weight_layer_f3c = Lambda((lambda x: x * connection_weight_f3c), name = "connection_weight_f3c")

    # define probabilties for failure
    dropout_fog_node_1 = constant(1 - survive_rates[0])
    dropout_fog_node_2 = constant(1 - survive_rates[1])
    dropout_fog_node_3 = constant(1 - survive_rates[2])

    # define lambda for fog failure
    failure_lambda = Lambda(lambda x : K.switch(K.variable(True), x * 0, x))
    
    # one input layer
    input_layer = Input(shape = (num_vars,))

    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="fog1_output_layer")(input_layer)
    #f1 = failure_lambda(f1)
    f1f2 = multiply_weight_layer_f1f2(f1)
    connection_f2 = Lambda(add_first_node_layers,name="F1_F2")(f1f2)
    #TODO: add branch lambda if there are multiple failures?

    # second fog node
    f2 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="fog2_input_layer")(connection_f2)
    f2 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="fog2_output_layer")(f2)
    #f2 = failure_lambda(f2)
    f1f3 = multiply_weight_layer_f1f3(f1)
    f2f3 = multiply_weight_layer_f2f3(f2)
    connection_f3 = Lambda(add_node_layers,name="F1F2_F3")([f1f3,f2f3])

    # third fog node
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="fog3_input_layer")(connection_f3)
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="fog3_layer_1")(f3)
    f3 = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="fog3_output_layer")(f3)
    f3 = failure_lambda(f3)
    f2c = multiply_weight_layer_f2c(f2)
    f3c = multiply_weight_layer_f3c(f3)
    connection_cloud = Lambda(add_node_layers,name="F2F3_FC")([f2c,f3c])

    # cloud node
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="cloud_input_layer")(connection_cloud)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="cloud_layer_1")(cloud)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="cloud_layer_2")(cloud)
    cloud = Dense(units=hidden_units,activation='relu',kernel_regularizer=regularizers.l1(regularization),name="cloud_layer_3")(cloud)

    # one output layer
    normal_output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)

    model = Model(inputs=input_layer, outputs=normal_output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# lambda function based on the probability will randonmly drop a node
# input: layer tensor and probability 
# output: outputs either a tensor of 0s with boolean indicating failure or the orginal tensor with a boolean indicating survival
def dropout_layer(input_tensor):
    layer = input_tensor[0]
    survive_prob = input_tensor[1]
    num = K.variable(np.random.random())
    # only dropout during training
    if K.learning_phase() == 0:
        return K.switch(K.greater(num,survive_prob),layer * 0, layer)
    else:
        return layer

def dropout_layer_output_shape(input_shape):
    return input_shape

# lambda function that does smart guessing based on training data when there is no data flow
# input: probabilties for each class
def smart_guessing(input_tensor):
    num = np.random.random()

# function that returns whether there is data flow in the network
def data_flow(failure_list):
    count = 0;
    for failure in failure_list:
        if failure:
            count+=1
    return count >= 2