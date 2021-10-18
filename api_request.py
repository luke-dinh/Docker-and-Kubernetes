import json, requests
import keras
from keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score, f1_score

# Load test set
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_test = keras.utils.to_categorical(y_test, num_classes=10)

