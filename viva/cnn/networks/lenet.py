from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


class LeNet:

    @staticmethod
    def build(width, height, depth, classes, weights_path=None):
        """build
        The LeNet architecture is as follows:
        INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

        :param width: the width of the input images
        :param height: the height of the input images
        :param depth: the depth (number of channels) of the input images
        :param classes: the number of class labels for the dataset
        :param weights_path: the path to the pre-trained model weights
        """
        model = Sequential()

        # first set CONV => RELU => POOL
        # The first layer will learn 20 convolution filters, each the size 5x5
        model.add(Conv2D(20, (5, 5), padding='same',
                         input_shape=(depth, height, width)))
        # ReLu activation function
        model.add(Activation('relu'))
        # 2x2 max-pooling in both x and y direction with a stride of 2
        # 2x2 window that slides across the activation layer and returns max value of each region
        # while taking a 2 pixel step in both x and y direction
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set CONV => RELU => POOL
        # 50 convolution filters
        # generally, convolutional layers deepen in layers as the image data is
        # abstracted into filters
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        # the output from the previous layer is flattened into a single vector
        model.add(Flatten())
        # the flattened vector is fully connected to 500 units
        model.add(Dense(500))
        model.add(Activation('relu'))

        # softmax classifier
        # fully connecting the previous layer into classes number of units
        model.add(Dense(classes))
        # return probabilities for each of our classes (multinomial logistic
        # regression)
        model.add(Activation('softmax'))

        # if the pre-trained weights have been supplied, load them
        if weights_path is not None:
            model.load_weights(weights_path)

        return model
