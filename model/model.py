import os 
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, InputLayer, BatchNormalization
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import fashion_mnist
import argparse

parser = argparse.ArgumentParser("Model Serving")
parser.add_argument("--input_shape", default=(28, 28, 1), type=list, help="Define input shape of model")
parser.add_argument("--save_path", default = "./serving/", type=str, help="Define path to save model")
parser.add_argument("--batch_size", default=64, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=20, type=int, help="Number of epochs")
opt = parser.parse_args()

input_shape = opt.input_shape
path = opt.save_path
num_epochs = opt.num_epochs
batch_size = opt.batch_size

def model(input_shape):

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    for num_filters in [16, 32, 64]:

        # Add conv layers
        model.add(Conv2D( 
            num_filters,
            kernel_size=(3,3),
            strides=1,
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

ex_model = model(input_shape=input_shape)

# Dataset
num_classes = 10
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Train and save weight
e_s = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
ex_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

ex_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.2, callbacks=[e_s])

# Evaluate
score = ex_model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test acc: ", score[1])

# Save model for serving
if not os.path.exists(path):
    os.makedirs(path)

version = "1"
export_path = os.path.join(path, version)
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model( 
    ex_model, 
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print("Model Saved!")