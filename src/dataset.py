import tensorflow.keras.layers as tfl
import tensorflow.keras 

def data_augmenter():
    data_augmentation=tensorflow.keras.Sequential()
    data_augmentation.add(tfl.RandomFlip("horizontal"))
    data_augmentation.add(tfl.RandomRotation(0.2))
    return data_augmentation

