from keras.models import Sequential
from keras.layers import Dense,Input,add,multiply,Lambda, BatchNormalization, Activation, Dropout
import keras.backend as K
import tensorflow as tf
from LambdaLayers import add_first_node_layers
from LambdaLayers import add_node_layers
from keras import regularizers
from keras import optimizers
from keras.models import Model

def define_baseline_functional_model(num_vars,num_classes,hidden_units,regularization):

    # dropout rate for all dropout layers 
    dropout = 0
    # one input layer
    input_layer = Input(shape = (num_vars,))
    # 10 hidden layers, 3 fog nodes
    # first fog node
    f1 = Dense(units=hidden_units,name="fog1_output_layer",activation='relu',kernel_initializer = 'he_normal')(input_layer)
    f1 = Dropout(dropout,seed=7)(f1)

    # second fog node
    f2 = Dense(units=hidden_units,name="fog2_input_layer",activation='relu',kernel_initializer = 'he_normal')(f1)
    f2 = Dropout(dropout,seed=7)(f2)
    f2 = Dense(units=hidden_units,name="fog2_output_layer",activation='relu',kernel_initializer = 'he_normal')(f2)
    f2 = Dropout(dropout,seed=7)(f2)

    # third fog node
    f3 = Dense(units=hidden_units,name="fog3_input_layer",activation='relu',kernel_initializer = 'he_normal')(f2)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_layer_1",activation='relu',kernel_initializer = 'he_normal')(f3)
    f3 = Dropout(dropout,seed=7)(f3)
    f3 = Dense(units=hidden_units,name="fog3_output_layer",activation='relu',kernel_initializer = 'he_normal')(f3)
    f3 = Dropout(dropout,seed=7)(f3)

    # cloud node
    cloud = Dense(units=hidden_units,name="cloud_input_layer",kernel_initializer = 'he_normal')(f3)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_1",kernel_initializer = 'he_normal')(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_2",kernel_initializer = 'he_normal')(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    cloud = Dense(units=hidden_units,name="cloud_layer_3",kernel_initializer = 'he_normal')(cloud)
    #cloud = BatchNormalization()(cloud)
    cloud = Activation(activation='relu')(cloud)
    cloud = Dropout(dropout,seed=7)(cloud)
    # one output layer
    output_layer = Dense(units=num_classes,activation='softmax',name = "output")(cloud)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model