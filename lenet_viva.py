from viva.cnn.networks.lenet import LeNet
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2

# command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--save-model', type=int, default=1,
                help='(optional) whether or not the model should be saved to disk')
ap.add_argument('-l', '--load-model', type=int, default=1,
                help='(optional) whether or not a pre-trained model should be loaded')
ap.add_argument('-w', '--weights', type=str,
                help='(optional) path to the weights file')
args = vars(ap.parse_args())

# TODO: fetch and preprocess data
# FYI The MNIST-LeNet data is 28 x 28 pixel image with channels [0, 1.0]

# Keras assumes at least 1 channel per image
# i.e. (number of images, 1, height, width)
print('[INFO] Preparing the data...')
dataset = None
(train_data, test_data, train_labels, test_labels) = (None, None, None, None)

# transforming the labels into a vector of 0 or 1 number of classes
# VIVA is either positive or negative so 2 classes
train_labels = np_utils.to_categorical(train_labels, 2)
test_labels = np_utils.to_categorical(test_labels, 2)

# compile the model
print('[INFO] Compiling the model...')
# optimizing with Stochastic Gradient Descent with a learning rate or 0.01
optimizer = SGD(lr=0.01)
# TODO: Figure out what the dimentions are for the input image
model = LeNet.build(width=28, height=28, depth=1, classes=2,
                    weights_path=args['weights_path'] if args['load_model'] > 0 else None)
# binary cross-entropy for loss function as theres only two classes
model.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

if args['load_model'] < 0:
    print('[INFO] Training...')
    model.fit(train_data, train_labels, batch_size=128, nb_epoch=20, verbose=1)

    print('[INFO] Evaluating...')
    (loss, accuracy) = model.evaluate(
        test_data, test_labels, batch_size=128, verbose=1)
    print('[INFO] Accuracy: {:.2f}%'.format(accuracy * 100))

if args['save_model'] > 0:
    print('[INFO] Dumping weights to file...')
    model.save_weights(args['weights'], overwrite=True)
