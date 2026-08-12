import tensorflow.keras as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
IMG_SIZE=(224,224,3)

base_model=MobileNetV2(include_top=False,input_shape=IMG_SIZE)
base_model.trainable=False
model=tf.models.Sequential([base_model,tfl.GlobalAveragePooling2D(), tfl.Dense(128,activation="relu"),tfl.Dense(1,activation="sigmoid")])



print(model.summary())