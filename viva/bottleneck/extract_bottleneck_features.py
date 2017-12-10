"""
Bottleneck Feature Extraction Functions
"""

from keras.applications.xception import Xception
import numpy as np


def extract_bottleneck_Xception(train_generator, valid_generator, test_generator,
                                img_shape, batch_size=20):
	"""extract_bottleneck_Xception
	Extracts bottleneck features from the Xception model with stock imagenet weights.
	The top is not included.

	:param train_generator: the training data generator
	:param valid_generator: the validation data generator
	:param test_generator: the test data generator
	:param img_shape: a tuple with the image width, height and number of channels
	:param num_samples: the number of samples
	:param batch_size: the size of the batch
	"""
    return extract_bottleneck(Xception(weights='imagenet', include_top=False), 
        input_shape=(img_shape[0], img_shape[1], img_shape[2]), 
        train_generator, valid_generator, test_generator, batch_size=batch_size)


"""
Helper Functions
"""


def extract_bottleneck(model, train_generator, valid_generator, test_generator,
                       img_shape, batch_size=20):
    train_features, train_labels = extract_features_labels(
        model, train_generator, num_samples, batch_size)
    valid_features, valid_labels = extract_features_labels(
        model, valid_generator, num_samples, batch_size)
    test_features, test_labels = extract_features_labels(
        model, test_generator, num_samples, batch_size)

    return {
        'train': {
            'features': train_features,
            'labels': train_labels
        },
        'valid': {
            'features': valid_features,
            'labels': valid_labels
        },
        'test': {
            'features': test_features,
            'labels': test_labels
        }
    }

# TODO: Number of samples is not passed through here. Can we get it with the generator? Otherwise we'll have to pass each train, valid, test number in

def extract_features_labels(model, generator, batch_size):
    """extract_bottleneck

    A function to extract bottleneck features from the pretrained Keras model.
    The shape of the outputs will vary but it will work if it is 4D tensor
    convolutional layer like this (none, 3, 3, 64).

    :param generator: data augmenting generator
    :param model: Keras CNN model without FC layers
    :param num_samples:
    :param batch_size:
    """

    """
    TODO: wrong batch size might give error with this right now.
    """

    # width = model.output_shape[1]
    # height = model.output_shape[2]
    # channel = model.output_shape[3]

    # temp_features = np.empty([batch_size, height, width, channel])
    # temp_labels = np.empty([batch_size, len(generator.class_indices)])

    # start = 0

    features, labels = next(generator)
    features = model.predict_on_batch(features)

    # while True:
    #     interval = start + batch_size
    #     if interval >= num_samples:
    #         interval = num_samples
    #     features, labels = generator.next()
    #     features = model.predict_on_batch(features)
    #     temp_features[start:interval] = features
    #     temp_labels[start:interval] = labels

    #     start = interval

    #     if interval == num_samples:
    #         break

    return features, labels
