import json, requests
import numpy as np
import keras
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Load test set
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_test = keras.utils.to_categorical(y_test, num_classes=10)

def show(idx, title):
  plt.figure()
  plt.imshow(x_test[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})


model_name='fashion_model'
host='localhost'
port=8502
signature_name="serving_default"


    # if imgs.ndim==3:
    #     imgs = np.expand_dims(imgs, axis=0)

data = json.dumps({ 
        "signature_name": signature_name,
        "instances": x_test[0:3].tolist()
    })

header = {"content-type": "application/json"}

json_response = requests.post( 
        'http://localhost:8502/v1/models/fashion_model:predict',
        data=data,
        headers=header
    )
y_pred = json.loads(json_response.text)
print(y_pred)

# show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
#         class_names[np.argmax(y_pred[0])], np.argmax(y_pred[0]), class_names[y_test[0]], y_test[0]))

# acc_score = accuracy_score(np.argmax(y_test, axis=-1), y_pred)
# f1 = f1_score(np.argmax(y_test, axis=-1), y_pred, average="macro")

# print("Accuracy Score: ", acc_score)
# print("F1 Score: ", f1)