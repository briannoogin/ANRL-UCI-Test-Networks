from keras.layers import Dense, Dropout, BatchNormalization, Activation

# modular fog block code
def fog_block(input,num_layers, hidden_units,dropout):
    for layer_index in range(num_layers):
        if layer_index == 0:
            output = Dense(units = hidden_units, activation= 'relu',kernel_initializer = 'he_normal')(input)
            output = Dropout(dropout)(output)
        else:
            output = Dense(units = hidden_units, activation= 'relu',kernel_initializer = 'he_normal')(output)
            output = Dropout(dropout)(output)
    return output

# modular cloud block code
def cloud_block(input,num_layers, hidden_units,dropout):
    for layer_index in range(num_layers):
        if layer_index == 0:
            output = Dense(units=hidden_units,name="cloud_input_layer",kernel_initializer = 'he_normal')(input)
            output = BatchNormalization()(output)
            output = Activation(activation='relu')(output)
        else:
            output = Dense(units=hidden_units,name="cloud_input_layer",kernel_initializer = 'he_normal')(output)
            output = BatchNormalization()(output)
            output = Activation(activation='relu')(output)
    return output