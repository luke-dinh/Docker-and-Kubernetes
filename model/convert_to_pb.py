import os
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.saved_model import builder, tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

# Disable eager mode
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

# Convert to model evaluation
tf.keras.backend.set_learning_phase(0) #Ignore dropout and inference
weight_model = load_model("./model/model.h5")
serving_path = "./serving/v1"

if not os.path.exists(serving_path):
    os.makedirs(serving_path)

builder = builder.SavedModelBuilder(serving_path)

signature = predict_signature_def( 
    inputs={ 
        'input_image': weight_model.inputs[0],
    },
    outputs={ 
        "Prediction": weight_model.outputs[0]
    }
)

with tf.compat.v1.keras.backend.get_session() as sess:
    builder.add_meta_graph_and_variables( 
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={'reid-predict': signature},
    )

    builder.save()