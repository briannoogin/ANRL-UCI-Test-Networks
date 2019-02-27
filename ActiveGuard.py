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
import random 

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

    # variables for active guard 
    f1_rand = K.variable(0)
    f2_rand = K.variable(0)
    f3_rand = K.variable(0)
    f1_survive_rate = K.variable(survive_rates[0])
    f2_survive_rate = K.variable(survive_rates[1])
    f3_survive_rate = K.variable(survive_rates[2])

    # set training phase to true 
    K.set_learning_phase(0)
    if K.learning_phase():
        # seeds so the random_number is different for each fog node 
        f1_rand = K.random_uniform(shape=f1_rand.shape,seed=7)
        f2_rand = K.random_uniform(shape=f2_rand.shape,seed=11)
        f3_rand = K.random_uniform(shape=f3_rand.shape,seed=42)
    # define lambda for fog failure, failures are only during training
    f1_failure_lambda = Lambda(lambda x : K.switch(K.greater(f1_rand,f1_survive_rate), x * 0, x),name = 'f1_failure_lambda')
    f2_failure_lambda = Lambda(lambda x : K.switch(K.greater(f2_rand,f2_survive_rate), x * 0, x),name = 'f2_failure_lambda')
    f3_failure_lambda = Lambda(lambda x : K.switch(K.greater(f3_rand,f3_survive_rate), x * 0, x),name = 'f3_failure_lambda')

    # one input layer
    input_layer = Input(shape = (num_vars,))

    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="fog1_output_layer",kernel_initializer = 'he_normal')(input_layer)
    f1 = Activation(activation='relu')(f1)
    #f1 = BatchNormalization()(f1)
    f1 = f1_failure_lambda(f1)
    f1f2 = multiply_weight_layer_f1f2(f1)
    duplicated_input = Dense(units=hidden_units,name="duplicated_input",activation='linear',kernel_initializer = 'he_normal')(input_layer)
    connection_f2 = Lambda(add_node_layers,name="F1_F2")([f1f2,duplicated_input])

    # second fog node
    f2 = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="fog2_input_layer",kernel_initializer = 'he_normal')(connection_f2)
    f2 = Activation(activation='relu')(f2)
    #f2 = BatchNormalization()(f2)
    f2 = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="fog2_output_layer",kernel_initializer = 'he_normal')(f2)
    f2 = Activation(activation='relu')(f2)
    #f2 = BatchNormalization()(f2)
    f2 = f2_failure_lambda(f2)
    f1f3 = multiply_weight_layer_f1f3(f1)
    f2f3 = multiply_weight_layer_f2f3(f2)
    connection_f3 = Lambda(add_node_layers,name="F1F2_F3")([f1f3,f2f3])

    # third fog node
    f3 = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="fog3_input_layer",kernel_initializer = 'he_normal')(connection_f3)
    f3 = Activation(activation='relu')(f3)
    #f3 = BatchNormalization()(f3)
    f3 = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l1(regularization),name="fog3_layer_1",kernel_initializer = 'he_normal')(f3)
    f3 = Activation(activation='relu')(f3)
    #f3 = BatchNormalization()(f3)
    f3 = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="fog3_output_layer",kernel_initializer = 'he_normal')(f3)
    f3 = Activation(activation='relu')(f3)
    #f3 = BatchNormalization()(f3)
    f3 = f3_failure_lambda(f3)
    f2c = multiply_weight_layer_f2c(f2)
    f3c = multiply_weight_layer_f3c(f3)
    connection_cloud = Lambda(add_node_layers,name="F2F3_FC")([f2c,f3c])

    # cloud node
    cloud = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="cloud_input_layer",kernel_initializer = 'he_normal')(connection_cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="cloud_layer_1",kernel_initializer = 'he_normal')(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="cloud_layer_2",kernel_initializer = 'he_normal')(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,activation='linear',kernel_regularizer=regularizers.l2(regularization),name="cloud_layer_3",kernel_initializer = 'he_normal')(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    # one output layer
    normal_output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)

    model = Model(inputs=input_layer, outputs=normal_output_layer)
    # TODO: define custom metric to keep track of network failing during training
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model