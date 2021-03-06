from keras.models import Sequential
from keras.layers import Dense,Input,Lambda, Activation
from KerasSingleLaneExperiment.LambdaLayers import add_node_layers
from keras.models import Model
import random
def define_deepFogGuard(num_vars,num_classes,hidden_units,survive_rates, skip_hyperconnections = [1,1,1],weight_config = 1):
    """Define a deepFogGuard model.
    ### Naming Convention
        ex: f2f1 = connection between fog node 2 and fog node 1
    ### Arguments
        num_vars (int): specifies number of variables from the data, used to determine input size.
        num_classes (int): specifies number of classes to be outputted by the model
        hidden_units (int): specifies number of hidden units per layer in network
        survive_rates (list): specifies the survival rate of each node in the network
        skip_hyperconnections (list): specifies the alive skip hyperconnections in the network, default value is [1,1,1]
        hyperconnection_weights (list): specifies the probability, default value is [1,1,1]
        weight_config (int): determines if the hyperconnections should be based on surivive_rates, 1: weighted 1, 2: weighted by weighted survival of multiple nodes, 3: weighted by survival of single node only, 4: weights are randomly weighted from 0-1, 5: weights are randomly weighted from 0-10
    ### Returns
        Keras Model object
    """

    if weight_config == 1:
        # all hyperconnection weights are weighted 1
        connection_weight_IoTf2  = 1
        connection_weight_ef2 = 1
        connection_weight_ef1 = 1
        connection_weight_f2f1 = 1
        connection_weight_f2c = 1
        connection_weight_f1c = 1
    elif weight_config == 2:
        # weights calculated by survival rates
        connection_weight_IoTf2  = 1 / (1+survive_rates[0])
        connection_weight_ef2 = survive_rates[0] / (1 + survive_rates[0])
        connection_weight_ef1 = survive_rates[0] / (survive_rates[0] + survive_rates[1])
        connection_weight_f2f1 = survive_rates[1] / (survive_rates[0] + survive_rates[1])
        connection_weight_f2c = survive_rates[1] / (survive_rates[1] + survive_rates[2])
        connection_weight_f1c = survive_rates[2] / (survive_rates[1] + survive_rates[2])
    elif weight_config == 3:
        # weighted by survival rates only, no divsion 
        connection_weight_IoTf2  = 1
        connection_weight_ef2 = survive_rates[0] 
        connection_weight_ef1 = survive_rates[0] 
        connection_weight_f2f1 = survive_rates[1]
        connection_weight_f2c = survive_rates[1] 
        connection_weight_f1c = survive_rates[2]
    elif weight_config == 4:
        #random.seed(42)
        # weights are randomly weighted from 0-1
        connection_weight_IoTf2  = random.uniform(0,1)
        connection_weight_ef2 = random.uniform(0,1)
        connection_weight_ef1 = random.uniform(0,1)
        connection_weight_f2f1 = random.uniform(0,1)
        connection_weight_f2c = random.uniform(0,1)
        connection_weight_f1c = random.uniform(0,1)
    elif weight_config == 5:
        #random.seed(42)
        # weights are randomly weighted from 0-10
        connection_weight_IoTf2  = random.uniform(0,10)
        connection_weight_ef2 = random.uniform(0,10)
        connection_weight_ef1 = random.uniform(0,10)
        connection_weight_f2f1 = random.uniform(0,10)
        connection_weight_f2c = random.uniform(0,10)
        connection_weight_f1c = random.uniform(0,10)
    elif weight_config == 6:
        # weights are weighted .5
        connection_weight_IoTf2  = .5
        connection_weight_ef2 = .5
        connection_weight_ef1 = .5
        connection_weight_f2f1 = .5
        connection_weight_f2c = .5
        connection_weight_f1c = .5
    else:
        raise ValueError("Invalid weight config value")

     # take away the skip hyperconnection if the value in hyperconnections array is 0
    if skip_hyperconnections[0] == 0:
        connection_weight_IoTf2 = 0
    if skip_hyperconnections[1] == 0:
        connection_weight_ef1 = 0
    if skip_hyperconnections[2] == 0:
        connection_weight_f2c = 0

    # define lambdas for multiplying node weights by connection weight
    multiply_weight_layer_IoTf2 = Lambda((lambda x: x * connection_weight_IoTf2), name = "connection_weight_IoTf2")
    multiply_weight_layer_ef2 = Lambda((lambda x: x * connection_weight_ef2), name = "connection_weight_ef2")
    multiply_weight_layer_ef1 = Lambda((lambda x: x * connection_weight_ef1), name = "connection_weight_ef1")
    multiply_weight_layer_f2f1 = Lambda((lambda x: x * connection_weight_f2f1), name = "connection_weight_f2f1")
    multiply_weight_layer_f2c = Lambda((lambda x: x * connection_weight_f2c), name = "connection_weight_f2c")
    multiply_weight_layer_f1c = Lambda((lambda x: x * connection_weight_f1c), name = "connection_weight_f1c")

   
    # IoT node
    IoT_node = Input(shape = (num_vars,))

    # edge node
    e = Dense(units=hidden_units,name="edge_output_layer",activation='relu')(IoT_node)
    ef2 = multiply_weight_layer_ef2(e)
    # use a linear Dense layer to transform input into the shape needed for the network
    duplicated_input = Dense(units=hidden_units,name="duplicated_input",activation='linear')(IoT_node)
    IoTf2 = multiply_weight_layer_IoTf2(duplicated_input)
    connection_f2 = Lambda(add_node_layers,name="F2_Input")([ef2,IoTf2])
 
    # fog node 2
    f2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu')(connection_f2)
    f2 = Dense(units=hidden_units,name="fog2_output_layer",activation='relu')(f2)
    f1f3 = multiply_weight_layer_ef1(e)
    f2f3 = multiply_weight_layer_f2f1(f2)
    connection_f1 = Lambda(add_node_layers,name="F1_Input")([f1f3,f2f3])

    # fog node 1
    f1 = Dense(units=hidden_units,name="fog1_input_layer",activation='relu')(connection_f1)
    f1 = Dense(units=hidden_units,name="fog1_layer_1",activation='relu')(f1)
    f1 = Dense(units=hidden_units,name="fog1_output_layer",activation='relu')(f1)
    f2c = multiply_weight_layer_f2c(f2)
    f1c = multiply_weight_layer_f1c(f1)
    connection_cloud = Lambda(add_node_layers,name="Cloud_Input")([f2c,f1c])

    # cloud node
    cloud = Dense(units=hidden_units,name="cloud_input_layer")(connection_cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_1")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2")(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3")(cloud)
    cloud = Activation(activation='relu')(cloud)
 
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    model = Model(inputs=IoT_node, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
