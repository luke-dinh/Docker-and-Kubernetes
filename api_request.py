import json, requests
import numpy as np
import keras
from keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score, f1_score

# Load test set
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_test = keras.utils.to_categorical(y_test, num_classes=10)

def res_infer( 
    imgs,
    model_name='mnist-serving',
    host='localhost',
    port=8501,
    signature_name="serving_default"
):

    if imgs.ndim==3:
        imgs = np.expand_dims(imgs, axis=0)

    data = json.dumps({ 
        "signature_name": signature_name,
        "instances": imgs.tolist()
    })

    header = {"content-type": "application/json"}