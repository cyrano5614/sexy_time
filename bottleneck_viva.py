from viva.cnn.networks.lenet import LeNet
from viva.cnn.networks.xception_transfer import Xception_Transfer
from read_viva import load_viva, DataGen
from pipeline import pretrained_model
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
ap.add_argument('-d', '--dataset-path', type=str, default='./data/detectiondata/',
                help='(optional) path to the VIVA dataset')
ap.add_argument('-e', '--epochs', type=int, default=20,
                help='(optional) number of epochs')
ap.add_argument('-b', '--batch-size', type=int, default=20,
                help='(optional) the batch size')
args = vars(ap.parse_args())

img_size = (128, 128)
batch_size = args['batch_size']


"""
Data Preprocessing
"""
print('[INFO] Preparing the data...')
path = args['dataset_path']
train_img_list, train_box_list, test_img_list, test_box_list = load_viva(path)
# Splitting into train validation 75/25
train_img_list, valid_img_list, train_box_list, valid_box_list = train_test_split(
    train_img_list, train_box_list, test_size=0.25)


"""
Model Compilation
"""
print('[INFO] Compiling the model...')
# optimizing with Stochastic Gradient Descent with a learning rate or 0.01
optimizer = SGD(lr=0.01)
# selecting model
model_name = args['model']

if model_name == 'xception':

    bottleneck_model = pretrained_model('Xception', img_size)

    train_generator = DataGen.generate_train(train_img_list, train_box_list,
                                     img_size=img_size, batch_size=batch_size,
                                     model=bottleneck_model,
                                     bottleneck=True, negative=True)
    valid_generator = DataGen(img_size, batch_size).generate_train(valid_img_list, valid_box_list,
                                     img_size=img_size, batch_size=batch_size,
                                     model=bottleneck_model,
                                     bottleneck=True, negative=True)
    test_generator = generate_batch(test_img_list, test_box_list,
                                    img_size=img_size, batch_size=batch_size,
                                    model=bottleneck_model,
                                    bottleneck=True, negative=True)

    x, y, z = bottleneck_model.output.shape[1:]

    model = Xception_Transfer.build((int(x), int(y), int(z)), 2)

else:

    model = LeNet.build(width=img_size[0], height=img_size[1], depth=3, classes=2,
                        weights_path=args['weights_path'] if args['load_model'] > 0 else None)

    train_generator = generate_batch(
        train_img_list, train_box_list, img_size=img_size, batch_size=batch_size, negative=True)
    valid_generator = generate_batch(
        valid_img_list, valid_box_list, img_size=img_size, batch_size=batch_size, negative=True)
    test_generator = generate_batch(
        test_img_list, test_box_list, img_size=img_size, batch_size=batch_size, negative=True)

# categorical cross-entropy for loss function
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])


"""
Model Fitting & Evaluating
"""
if args['load_model'] < 1:
    print('[INFO] Training...')
    model.fit_generator(train_generator, steps_per_epoch=len(train_img_list)//batch_size, epochs=args['epochs'],
                        validation_data=valid_generator, validation_steps=len(valid_img_list)//batch_size, verbose=1)

    print('[INFO] Evaluating...')
    (loss, accuracy) = model.evaluate_generator(
        test_generator, steps=len(test_img_list)//batch_size)
    print('[INFO] Accuracy: {:.2f}%'.format(accuracy * 100))

if args['save_model'] > 0:
    print('[INFO] Dumping weights to file...')
    model.save_weights(args['weights_path'], overwrite=True)
