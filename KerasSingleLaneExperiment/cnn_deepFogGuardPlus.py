
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

import keras.backend as K
import keras.layers as layers
from keras.backend import zeros
from keras_applications.imagenet_utils import _obtain_input_shape, get_submodules_from_kwargs
from keras_applications import imagenet_utils
import keras 
from KerasSingleLaneExperiment.cnn_utils import _conv_block,_depthwise_conv_block

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')

def define_deepFogGuardPlus_CNN(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000, 
              survive_rates = [1,1],
              **kwargs):
    """Instantiates the MobileNet architecture.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            width multiplier in the MobileNet paper.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution. This
            is called the resolution multiplier in the MobileNet paper.
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        survive_rates: survival rates of network nodes, default value is [1,1]
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    backend,layers,_, keras_utils = get_submodules_from_kwargs(kwargs)
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')
    backend = keras.backend
    layers = keras.layers
    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not in [128, 160, 192, 224]. '
                          'Weights for input shape (224, 224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # variables for node dropout
    edge_rand = K.variable(0)
    fog_rand = K.variable(0)
    edge_survive_rate = K.variable(survive_rates[0])
    fog_survive_rate = K.variable(survive_rates[1])
    # set training phase to true 
    if K.eval(K.learning_phase()):
        # seeds so the random_number is different for each fog node 
        edge_rand = K.random_uniform(shape=edge_rand.shape,seed=7)
        fog_rand = K.random_uniform(shape=fog_rand.shape,seed=11)
     # define lambda for failure, only fail during training
    edge_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(edge_rand,edge_survive_rate), x * 0, x),name = 'edge_failure_lambda')
    fog_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(fog_rand,fog_survive_rate), x * 0, x),name = 'fog_failure_lambda')
   
    # changed the strides from 2 to 1 since cifar-10 images are smaller
    # IoT node
    iot = _conv_block(img_input, 32, alpha, strides=(1, 1)) # size: (31,31,16)
    connection_iotfog = layers.Conv2D(64,(1,1),strides = 1, use_bias = False, name = "skip_hyperconnection_iotfog")(iot)
 
    # edge 
    edge = _depthwise_conv_block(iot, 64, alpha, depth_multiplier, block_id=1)

    edge = _depthwise_conv_block(edge, 128, alpha, depth_multiplier,
                              strides=(1, 1), block_id=2)
    connection_edgefog = _depthwise_conv_block(edge, 128, alpha, depth_multiplier, block_id=3) # size:  (None, 31, 31, 64) 
    connection_edgefog = edge_failure_lambda(connection_edgefog)
    # skip hyperconnection, used 1x1 convolution to project shape of node output into (7,7,256)\
    # check it back to normal skip_hyperconnection
    connection_edgecloud = layers.Conv2D(256,(1,1),strides = 4, use_bias = False, name = "skip_hyperconnection_edgecloud")(connection_edgefog)
    connection_fog = layers.add([connection_iotfog,connection_edgefog], name = "connection_fog")
    
    # fog node
    fog = _depthwise_conv_block(connection_fog, 256, alpha, depth_multiplier, # size: (None, 32, 32, 64)
                              strides=(2, 2), block_id=4)
    fog = _depthwise_conv_block(fog, 256, alpha, depth_multiplier, block_id=5)

    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=7)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=8) #size : (None, 7, 7, 256) 
    # pad from (7,7,256) to (8,8,256)
    connection_fogcloud = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)), name = "fogcloud_connection_padding")(fog)
    connection_fogcloud = fog_failure_lambda(connection_fogcloud)
    connection_cloud = layers.add([connection_fogcloud,connection_edgecloud], name = "connection_cloud")

    # cloud node
    cloud = _depthwise_conv_block(connection_cloud, 512, alpha, depth_multiplier, block_id=9) # size: (None, 7, 7, 256)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=10)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=11)

    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if backend.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        cloud = layers.GlobalAveragePooling2D()(cloud)
        cloud = layers.Reshape(shape, name='reshape_1')(cloud)
        cloud = layers.Conv2D(classes, (1, 1),
                          padding='same',
                          name='conv_preds')(cloud)
        cloud = layers.Reshape((classes,), name='reshape_2')(cloud)
        cloud = layers.Activation('softmax', name='act_softmax')(cloud)
    else:
        if pooling == 'avg':
            cloud = layers.GlobalAveragePooling2D()(cloud)
        elif pooling == 'max':
            cloud = layers.GlobalMaxPooling2D()(cloud)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = keras.Model(inputs, cloud, name='mobilenet_%0.2f_%s' % (alpha, rows))

    # Load weights.
    if weights == 'imagenet':
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras_utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras_utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def define_adjusted_deepFogGuardPlus_CNN(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000, 
              survive_rates = [1,1],
              **kwargs):
    """Instantiates the MobileNet architecture.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            width multiplier in the MobileNet paper.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution. This
            is called the resolution multiplier in the MobileNet paper.
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        survive_rates: survival rates of network nodes, default value is [1,1]
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    backend,layers,_, keras_utils = get_submodules_from_kwargs(kwargs)
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')
    backend = keras.backend
    layers = keras.layers
    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not in [128, 160, 192, 224]. '
                          'Weights for input shape (224, 224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # variables for node dropout
    edge_rand = K.variable(0)
    fog_rand = K.variable(0)
    edge_survive_rate = K.variable(survive_rates[0])
    fog_survive_rate = K.variable(survive_rates[1])
    # set training phase to true 
    if K.eval(K.learning_phase()):
        # seeds so the random_number is different for each fog node 
        edge_rand = K.random_uniform(shape=edge_rand.shape,seed=7)
        fog_rand = K.random_uniform(shape=fog_rand.shape,seed=11)
     # define lambda for failure, only fail during training
    edge_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(edge_rand,edge_survive_rate), x * 0, x),name = 'edge_failure_lambda')
    fog_failure_lambda = layers.Lambda(lambda x : K.switch(K.greater(fog_rand,fog_survive_rate), x * 0, x),name = 'fog_failure_lambda')
        
    # define lambda for adjusting weights
    ef_dropout_multiply = layers.Lambda(lambda x : K.switch(K.learning_phase(), x, x * edge_survive_rate),name = 'ef_dropout_lambda')
    ec_dropout_multiply = layers.Lambda(lambda x : K.switch(K.learning_phase(), x, x * edge_survive_rate),name = 'ec_dropout_lambda')
    fc_dropout_multiply = layers.Lambda(lambda x : K.switch(K.learning_phase(),x, x * fog_survive_rate),name = 'fc_dropout_lambda')
    # changed the strides from 2 to 1 since cifar-10 images are smaller
    # IoT node
    iot = _conv_block(img_input, 32, alpha, strides=(1, 1)) # size: (31,31,16)
    connection_iotfog = layers.Conv2D(64,(1,1),strides = 1, use_bias = False, name = "skip_hyperconnection_iotfog")(iot)
 
    # edge 
    edge = _depthwise_conv_block(iot, 64, alpha, depth_multiplier, block_id=1)

    edge = _depthwise_conv_block(edge, 128, alpha, depth_multiplier,
                              strides=(1, 1), block_id=2)
    connection_edgefog = _depthwise_conv_block(edge, 128, alpha, depth_multiplier, block_id=3) # size:  (None, 31, 31, 64) 
    connection_edgefog = edge_failure_lambda(connection_edgefog)
    connection_edgefog = ef_dropout_multiply(connection_edgefog)
    # skip hyperconnection, used 1x1 convolution to project shape of node output into (7,7,256)\
    # check it back to normal skip_hyperconnection
    connection_edgecloud = layers.Conv2D(256,(1,1),strides = 4, use_bias = False, name = "skip_hyperconnection_edgecloud")(connection_edgefog)
    connection_edgecloud = ec_dropout_multiply(connection_edgecloud)
    connection_fog = layers.add([connection_iotfog,connection_edgefog], name = "connection_fog")
    
    # fog node
    fog = _depthwise_conv_block(connection_fog, 256, alpha, depth_multiplier, # size: (None, 32, 32, 64)
                              strides=(2, 2), block_id=4)
    fog = _depthwise_conv_block(fog, 256, alpha, depth_multiplier, block_id=5)

    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=7)
    fog = _depthwise_conv_block(fog, 512, alpha, depth_multiplier, block_id=8) #size : (None, 7, 7, 256) 
    # pad from (7,7,256) to (8,8,256)
    connection_fogcloud = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)), name = "fogcloud_connection_padding")(fog)
    connection_fogcloud = fog_failure_lambda(connection_fogcloud)
    connection_fogcloud = fc_dropout_multiply(connection_fogcloud)
    connection_cloud = layers.add([connection_fogcloud,connection_edgecloud], name = "connection_cloud")

    # cloud node
    cloud = _depthwise_conv_block(connection_cloud, 512, alpha, depth_multiplier, block_id=9) # size: (None, 7, 7, 256)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=10)
    cloud = _depthwise_conv_block(cloud, 512, alpha, depth_multiplier, block_id=11)

    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    cloud = _depthwise_conv_block(cloud, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if backend.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        cloud = layers.GlobalAveragePooling2D()(cloud)
        cloud = layers.Reshape(shape, name='reshape_1')(cloud)
        cloud = layers.Conv2D(classes, (1, 1),
                          padding='same',
                          name='conv_preds')(cloud)
        cloud = layers.Reshape((classes,), name='reshape_2')(cloud)
        cloud = layers.Activation('softmax', name='act_softmax')(cloud)
    else:
        if pooling == 'avg':
            cloud = layers.GlobalAveragePooling2D()(cloud)
        elif pooling == 'max':
            cloud = layers.GlobalMaxPooling2D()(cloud)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = keras.Model(inputs, cloud, name='mobilenet_%0.2f_%s' % (alpha, rows))

    # Load weights.
    if weights == 'imagenet':
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras_utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras_utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model