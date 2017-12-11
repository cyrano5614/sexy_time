from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense
import numpy as np


class Xception_Transfer:

    @staticmethod
    def build(shape, num_labels):
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=shape))
        model.add(Dense(2, activation='softmax'))

        return model
