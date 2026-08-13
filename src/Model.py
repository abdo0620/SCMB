import tensorflow.keras as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

import dataset
IMG_SIZE=(224,224,3)
base_model=MobileNetV2(include_top=False,input_shape=IMG_SIZE,weights="imagenet")
base_model.trainable=False

print(len(base_model.layers))
print(base_model.summary())
def detect(img_size=IMG_SIZE, augmenter=dataset.data_augmenter):
    model=tf.models.Sequential([augmenter(),tfl.Lambda(preprocess_input),base_model,tfl.GlobalAveragePooling2D(), tfl.Dense(128,activation="relu"),tfl.Dense(1,activation="sigmoid")])
    return model 
model=detect()

print(len(model.layers))
print(model.summary())
print(((model.weights)))