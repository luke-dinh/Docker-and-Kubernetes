import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Input, BatchNormalization
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import mnist
import argparse

parser = argparse.ArgumentParser("Model Serving")
parser.add_argument("--input_shape", default=[28, 28, 1], type=list, help="Define input shape of model")
parser.add_argument("--svae_path", default = "model/", type=str, help="Define path to save model")
opt = parser.parse_args()

input_shape = opt.input_shape
path = opt.save_path

def model(input_shape):

    model = Sequential()
    model.add(Input(shape=input_shape))

    for num_filters in [16, 32, 64]:

        # Add conv layers
        model.add(Conv2D( 
            num_filters,
            kernel_size=(3,3),
            stride=1,
            padding='same',
            activation='relu'
        ))

        # Add MaxPooling2D
        model.add(MaxPool2D(pool_size=(2,2), padding='same'))

        # Add Batchnorm
        model.add(BatchNormalization(axis=-1))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    return model