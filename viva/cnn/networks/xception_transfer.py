from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense


class Xception_Transfer:

    @staticmethod
    def build(train_features, train_labels):
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
        model.add(Dense(len(np.unique(train_labels, axis=0)), activation='softmax'))

        return model
