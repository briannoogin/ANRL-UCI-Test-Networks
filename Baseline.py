from keras.models import Sequential
from keras.layers import Dense,Input,add,multiply,Lambda, BatchNormalization, Activation
import keras.backend as K
import tensorflow as tf
from LambdaLayers import add_first_node_layers
from LambdaLayers import add_node_layers
from keras import regularizers
from keras import optimizers
from keras.models import Model

def define_baseline_functional_model(num_vars,num_classes,hidden_units, regularization):
   # one input layer
    input_layer = Input(shape = (num_vars,))

    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="fog1_output_layer")(input_layer)
    f1 = BatchNormalization()(f1)
    f1 = Activation(activation='relu')(f1)
    connection_f1 = Lambda(add_first_node_layers,name="F1_F2")(f1)
    # second fog node
    f2 = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="fog2_input_layer")(connection_f1)
    f2 = BatchNormalization()(f2)
    f2 = Activation(activation='relu')(f2)
    f2 = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="fog2_output_layer")(f2)
    f2 = BatchNormalization()(f2)
    f2 = Activation(activation='relu')(f2)
    connection_f2 = Lambda(add_first_node_layers,name="F1F2_F3")(f2)
    # third fog node
    f3 = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="fog3_input_layer")(connection_f2)
    f3 = BatchNormalization()(f3)
    f3 = Activation(activation='relu')(f3)
    f3 = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="fog3_layer_1")(f3)
    f3 = BatchNormalization()(f3)
    f3 = Activation(activation='relu')(f3)
    f3 = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="fog3_output_layer")(f3)
    f3 = BatchNormalization()(f3)
    f3 = Activation(activation='relu')(f3)
    connection_f3 = Lambda(add_first_node_layers,name="F2F3_FC")(f2)
    # cloud node
    cloud = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="cloud_input_layer")(connection_f3)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="cloud_layer_1")(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="cloud_layer_2")(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dense(units=hidden_units,kernel_regularizer=regularizers.l1(regularization),name="cloud_layer_3")(cloud)
    cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    # TODO: make a lambda functoin that checks if there is no data connection flow and does smart random guessing
    model = Model(inputs=input_layer, outputs=output_layer)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model