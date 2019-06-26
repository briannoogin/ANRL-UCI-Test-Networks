import keras
from keras.datasets import cifar10
from keras.applications.mobilenet_v2 import MobileNetV2
if __name__ == "__main__":
    # get cifar10 data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    model = MobileNetV2(weights = None,classes=10,input_shape = (32,32,3))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train,y_train, batch_size=32,verbose=True,shuffle = True, epoch = 10)    
    print(model.evaluate(x_test,y_test))