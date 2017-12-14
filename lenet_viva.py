from viva.cnn.networks.lenet import LeNet
from viva.cnn.networks.xception_transfer import Xception_Transfer
from read_viva import load_viva, DataGen, plot_model_history
from pipeline import pretrained_model
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
print('[INFO] Loaded {} training images'.format(len(train_img_list)))
print('[INFO] Loaded {} validation images'.format(len(valid_img_list)))
print('[INFO] Loaded {} testing images'.format(len(test_img_list)))


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
    bottleneck_model._make_predict_function()

    train_generator = DataGen(img_size,
                              batch_size,
                              negative=True,
                              bottleneck=True,
                              model=bottleneck_model).generate_train(train_img_list, train_box_list)
    valid_generator = DataGen(img_size,
                              batch_size,
                              negative=True,
                              bottleneck=True,
                              model=bottleneck_model).generate_train(valid_img_list, valid_box_list)
    test_generator = DataGen(img_size,
                              batch_size,
                              negative=True,
                              bottleneck=True,
                              model=bottleneck_model).generate_train(test_img_list, test_box_list)

    x, y, z = bottleneck_model.output.shape[1:]

    model = Xception_Transfer.build((int(x), int(y), int(z)), 2)

else:

    model = LeNet.build(width=img_size[0], height=img_size[1], depth=3, classes=2,
                        weights_path=args['weights_path'] if args['load_model'] > 0 else None)

    train_generator = DataGen(img_size,
                              batch_size,
                              negative=True).generate_train(train_img_list,
                                                                   train_box_list)
    valid_generator = DataGen(img_size,
                              batch_size,
                              negative=True).generate_train(valid_img_list,
                                                                   valid_box_list)
    test_generator = DataGen(img_size,
                             batch_size,
                             negative=True,
                             verbose=0).generate_train(test_img_list,
                                                                   test_box_list)

# categorical cross-entropy for loss function
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])


"""
Model Fitting & Evaluating
"""
if args['load_model'] < 1:

    print('[INFO] Training...')

    checkpointer = ModelCheckpoint(filepath='./models/weights.best.{}.hdf5'.format(model.name),
                                   verbose=1, save_best_only=True)

    early = EarlyStopping(monitor='val_acc',
                          min_delta=0,
                          patience=10,
                          verbose=1)

    callback_list = [checkpointer, early]

    model_info = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_img_list)//batch_size,
                                     epochs=args['epochs'],
                                     validation_data=valid_generator,
                                     validation_steps=len(valid_img_list)//batch_size,
                                     callbacks=callback_list,
                                     verbose=1)

    plot_model_history(model_info)

    print('[INFO] Evaluating...')

    (loss, accuracy) = model.evaluate_generator(test_generator,
                                                10)

    print('[INFO] Accuracy: {:.2f}%'.format(accuracy * 100))
