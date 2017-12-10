from viva.cnn.networks.lenet import LeNet
from viva.cnn.networks.xception_transfer import Xception_Transfer
from read_viva import load_viva, generate_batch
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import cv2


"""
Command Line Arguments
"""
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, default='lenet',
                help='(optional) select the model to train')
ap.add_argument('-s', '--save-model', type=int, default=1,
                help='(optional) whether or not the model should be saved to disk')
ap.add_argument('-l', '--load-model', type=int, default=0,
                help='(optional) whether or not a pre-trained model should be loaded')
ap.add_argument('-w', '--weights-path', type=str, default='models/viva_weights.hdf5',
                help='(optional) path to the weights file')
ap.add_argument('-d', '--dataset-path', type=str, default='./data/detectiondata/train/',
                help='(optional) path to the VIVA dataset')
ap.add_argument('-e', '--epochs', type=int, default=20,
                help='(optional) number of epochs')
ap.add_argument('-b', '--batch-size', type=int, default=20,
                help='(optional) the batch size')
args = vars(ap.parse_args())

img_size = (224, 224)
batch_size = args['batch_size']

"""
Data Preprocessing
"""
print('[INFO] Preparing the data...')
path = args['dataset_path']
img_list, box_list = load_viva(path)
# Splitting into 60/20/20
train_imgs, test_imgs, train_boxes, test_boxes = train_test_split(
    img_list, box_list, test_size=0.2)
train_imgs, valid_imgs, train_boxes, valid_boxes = train_test_split(
    train_imgs, train_boxes, test_size=0.25)

train_generator = generate_batch(
    train_imgs, train_boxes, img_size=img_size, batch_size=batch_size)
valid_generator = generate_batch(
    valid_imgs, valid_boxes, img_size=img_size, batch_size=batch_size)
test_generator = generate_batch(
    test_imgs, test_boxes, img_size=img_size, batch_size=batch_size)


"""
Model Compilation
"""
print('[INFO] Compiling the model...')
# optimizing with Stochastic Gradient Descent with a learning rate or 0.01
optimizer = SGD(lr=0.01)
# selecting model
model_name = args['model']
if model_name == 'lenet':
    model = LeNet.build(width=img_size[0], height=img_size[1], depth=3, classes=2,
                        weights_path=args['weights_path'] if args['load_model'] > 0 else None)
# elif model_name == 'xception':
#     bottleneck_features = extract_bottleneck_Xception(train_generator, valid_generator, test_generator,
#                                                       img_shape=(img_size[0], img_size[1], 3), num_samples=5000, batch_size=batch_size)
#     model = Xception_Transfer.build(bottleneck_features['train']['features'],
#                                     bottleneck_features['train']['labels'])
else:
    model = LeNet.build(width=img_size[0], height=img_size[1], depth=3, classes=2,
                        weights_path=args['weights_path'] if args['load_model'] > 0 else None)
# categorical cross-entropy for loss function
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])


"""
Model Fitting & Evaluating
"""
if args['load_model'] < 1:
    print('[INFO] Training...')
    model.fit_generator(train_generator, steps_per_epoch=len(train_imgs)//batch_size, epochs=args['epochs'],
                        validation_data=valid_generator, validation_steps=len(valid_imgs)//batch_size, verbose=1)

    print('[INFO] Evaluating...')
    (loss, accuracy) = model.evaluate_generator(test_generator, verbose=1)
    print('[INFO] Accuracy: {:.2f}%'.format(accuracy * 100))

if args['save_model'] > 0:
    print('[INFO] Dumping weights to file...')
    model.save_weights(args['weights_path'], overwrite=True)
