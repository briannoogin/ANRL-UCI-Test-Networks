import keras
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
import math
# cnn experiment 
if __name__ == "__main__":
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize input
    x_train = x_train / 255
    x_test = x_test / 255
    datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    )
    model = MobileNet(weights = None,classes=10,input_shape = (32,32,3),dropout = .5)
 
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.fit(x_train,y_train, batch_size=128,verbose=True,shuffle = True, epochs = 25)    
    num_samples = len(x_train)
    batch_size = 128
    steps_per_epoch = math.ceil(num_samples / batch_size)
    model.fit_generator(datagen.flow(x_train,y_train,batch_size = batch_size),epochs = 50,validation_data = (x_test,y_test), steps_per_epoch = steps_per_epoch)
    print(model.evaluate(x_test,y_test))