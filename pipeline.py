import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Activation, GlobalAveragePooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.datasets import load_files
from keras.preprocessing import image
from tqdm import tqdm

def generate_train_directory(path, img_size, batch_size):

    # Specify Augmenting options
    train_augmentor = ImageDataGenerator(width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         horizontal_flip=True,
                                         rotation_range=3,
                                         shear_range=0.2,
                                         vertical_flip=True,
                                         rescale=1./255,
                                         zoom_range=[.2, .2])

    # train_augmentor = ImageDataGenerator(rescale=1./255)

    train_generator = train_augmentor.flow_from_directory(path,
                                                          target_size=img_size,
                                                          batch_size=batch_size)
                                                          # class_mode=None,
                                                          # save_to_dir='./augmented_training/',
                                                          # shuffle=False)


    return train_generator

def generate_test_directory(path, img_size, batch_size):

    test_augmentor = ImageDataGenerator(rescale=1./255)

    test_generator = test_augmentor.flow_from_directory(path,
                                                        target_size=img_size,
                                                        batch_size=batch_size)
                                                        # class_mode=None,
                                                        # shuffle=False)

    return test_generator


def load_data(path):
    data = load_files(path)
    data_files = data['filenames']
    data_targets = np_utils.to_categorical(data['target'])
    return data_files, data_targets


def path_to_tensor(img_path, img_size=(128, 128)):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def pretrained_model(model_name, img_size):
    """pretrained_model

    A function to get popular CNN architectures using Keras.  All the final
    fully connnected layers have been removed to add our own classification
    layer.

    :param model_name:
    :param img_size: set image width and height in tuple like this (128,128)
                     this image size needs to be same as our data output.


    """

    if model_name == 'VGG16':
        from keras.applications.vgg16 import VGG16
        return VGG16(weights='imagenet', include_top=False,
                     input_shape=(img_size[0], img_size[1], 3))

    elif model_name == 'VGG19':
        from keras.applications.vgg19 import VGG19
        return VGG19(weights='imagenet', include_top=False,
                     input_shape=(img_size[0], img_size[1], 3))

    elif model_name == 'ResNet50':
        from keras.applications.resnet50 import ResNet50
        return ResNet50(weights='imagenet', include_top=False,
                        input_shape=(img_size[0], img_size[1], 3))

    elif model_name == 'Xception':
        from keras.applications.xception import Xception
        return Xception(weights='imagenet', include_top=False,
                        input_shape=(img_size[0], img_size[1], 3))

    elif model_name == 'InceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        return InceptionV3(weights='imagenet', include_top=False,
                           input_shape=(img_size[0], img_size[1], 3))

    else:
        raise ValueError("Don't do it....not yet")
        # return model_name


def extract_bottleneck(generator, model, num_samples, batch_size):
    """extract_bottleneck

    A function to extract bottleneck features from the pretrained Keras model.
    The shape of the outputs will vary but it will work if it is 4D tensor
    convolutional layer like this (none, 3, 3, 64).

    :param generator: data augmenting generator
    :param model: Keras CNN model without FC layers
    :param num_samples:
    :param batch_size:
    """

    width = model.output_shape[1]
    height = model.output_shape[2]
    channel = model.output_shape[3]

    temp_features = np.empty([num_samples, height, width, channel])
    temp_labels = np.empty([num_samples, len(generator.class_indices)])

    start = 0

    while True:
        interval = start + batch_size
        if interval >= num_samples:
            interval = num_samples
        features, labels = generator.next()
        features = model.predict_on_batch(features)
        temp_features[start:interval] = features
        temp_labels[start:interval] = labels

        start = interval

        if interval == num_samples:
            break

    return temp_features, temp_labels

def generate_bottleneck(model_name, img_path, batch_size = 20, num_samples = 7000, augment=True, img_size=(128, 128)):

    train_datagen = generate_train_directory(img_path,
                                             img_size=img_size,
                                             batch_size=batch_size)

    valid_datagen = generate_test_directory(img_path,
                                            img_size=img_size,
                                            batch_size=batch_size)

    test_datagen = generate_test_directory(img_path,
                                           img_size=img_size,
                                           batch_size=batch_size)

    base_model = pretrained_model(model_name=model_name, img_size=img_size)

    # Not neceassry right now but future customization
    # for layer in base_model.layers:
    #     layer.trainable = False

    train_size = num_samples
    valid_size = int(num_samples * 0.2)
    test_size = int(num_samples * 0.2)

    train_features, train_labels = extract_bottleneck(train_datagen,
                                                      model=base_model,
                                                      num_samples=train_size,
                                                      batch_size=batch_size)

    valid_features, valid_labels = extract_bottleneck(valid_datagen,
                                                      model=base_model,
                                                      num_samples=valid_size,
                                                      batch_size=batch_size)

    test_features, test_labels = extract_bottleneck(valid_datagen,
                                                    model=base_model,
                                                    num_samples=test_size,
                                                    batch_size=batch_size)

    output = {'train_features': train_features,
              'train_labels': train_labels,
              'valid_features': valid_features,
              'valid_labels': valid_labels,
              'test_features': test_features,
              'test_labels': test_labels}

    return output


def bottleneck_model(data):

    # base_model = pretrained_model(model_name=model_name, img_size=img_size)

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=data['train_features'].shape[1:]))
    model.add(Dense(len(np.unique(data['train_labels'], axis=0)), activation='softmax'))

    # model.summary()

    return model


def train_model(model, data, name, epochs=10, batch_size=20):

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='./models/weights.best.{}.hdf5'.format(name),
                                   verbose=1, save_best_only=True)

    early = EarlyStopping(monitor='val_acc',
                          min_delta=0,
                          patience=10,
                          verbose=1)

    if 'weights.best.{}.hdf5'.format(name) in os.listdir('./models/'):
        model.load_weights('models/weights.best.{}.hdf5'.format(name))
    else:
        model.fit(data['train_features'],
                  data['train_labels'],
                  validation_data=(data['valid_features'],
                                   data['valid_labels']),
                  epochs=epochs,
                  batch_size=20,
                  callbacks=[checkpointer, early],
                  verbose=1)

        model.load_weights('models/weights.best.{}.hdf5'.format(name))

    return model


def predict_model(model, data):

    predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in data['test_features']]

    test_accuracy = 100 * np.sum(np.array(predictions)==np.argmax(data['test_labels'], axis=1))/len(predictions)

    print('Test accuracy: {:.4f}'.format(test_accuracy))


def quick_cum(model_name, path, name):

    data = generate_bottleneck(model_name, path)

    model = bottleneck_model(data)

    model = train_model(model, data, name)

    predict_model(model, data)


