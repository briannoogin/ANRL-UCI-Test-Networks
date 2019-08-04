from KerasSingleLaneExperiment.cnn import define_deepFogGuardPlus_CNN
model = define_deepFogGuardPlus_CNN(weights = None,classes=10,input_shape = (32,32,3),dropout = 0, alpha = .5)
layer = model.get_layer(index = 0)
print(len(layer.input_shape))
