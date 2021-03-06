"""
Helper functions for the VIVA hand dataset.
"""

import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from random import random
from keras.utils import np_utils
from keras.applications.xception import preprocess_input
from time import sleep


class Point(object):
    """Point"""

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rect(object):
    """Rect"""

    def __init__(self, p1, p2):
        '''Store the top, bottom, left and right values for points
               p1 and p2 are the (corners) in either order
        '''

        self.left = min(p1.x, p2.x)
        self.right = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top = max(p1.y, p2.y)


def load_viva(path):
    """load_viva
    Loads VIVA dataset file names sorted into list.
    Returns img and box list.

    :param path: train or test directory path
    """

    train_img_path = path + 'train/pos/'
    train_box_path = path + 'train/posGt/'

    train_img_list = sorted(glob.glob(train_img_path + '*'))
    train_box_list = sorted(glob.glob(train_box_path + '*'))

    test_img_path = path + 'test/pos/'
    test_box_path = path + 'test/posGt/'

    test_img_list = sorted(glob.glob(test_img_path + '*'))
    test_box_list = sorted(glob.glob(test_box_path + '*'))

    return train_img_list, train_box_list, test_img_list, test_box_list


def extract_box(path):
    """extract_box
    Extract annotation box positions for each labels from VIVA hand dataset.
    output is a list of tuples.

    :param path: text file path
    """

    with open(path) as temp:
        output = []

        for i, line in enumerate(temp):

            if i != 0 and line:
                label, x_1, y_1, x_off, y_off, *_ = line.split()
                pt_1 = (int(x_1), int(y_1))
                pt_2 = (pt_1[0] + int(x_off), (pt_1[1] + int(y_off)))
                output.append((label, pt_1, pt_2))

    return output


def visualize_box(img, boxes):
    """visualize_box
    Takes input parameters and outputs a drawn image.  Does not alter
    original image.

    :param img: numpy array like image
    :param boxes: list of label and two box corners in a tuple
                  example = [('label', (123, 123), (234, 234))]
    """

    out_img = img.copy()

    for box in boxes:

        cv2.rectangle(out_img, box[1], box[2], (255, 255, 0), 3)
        cv2.putText(out_img, box[0], box[1],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (200, 255, 155), 4, cv2.LINE_AA)

    return out_img


def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    return (range_overlap(r1.left, r1.right, r2.left, r2.right) and
            range_overlap(r1.bottom, r1.top, r2.bottom, r2.top))


def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''

    return (a_min <= b_max) and (b_min <= a_max)


def crop_images(img_path, box_path, img_size, negative=False):
    """crop_images
    Takes input image and box coordinates and returns cropped out images

    :param img: original image path we will extract from
    :param boxes: annotation box coordinates from annotation file
    """

    img = mpimg.imread(img_path)
    boxes = extract_box(box_path)

    req_numb = len(boxes)

    out_images = []
    out_labels = []

    x_off = img_size[0]
    y_off = img_size[1]

    if negative == True:

        while len(out_images) < req_numb:

            x_1 = np.random.randint(0, img.shape[0] - x_off)
            y_1 = np.random.randint(0, img.shape[1] - y_off)

            x_2 = x_1 + x_off
            y_2 = y_1 + y_off

            temp_rect = Rect(Point(x_1, y_1), Point(x_2, y_2))

            for box in boxes:

                p1 = Point(box[1][0], box[1][1])
                p2 = Point(box[2][0], box[2][1])
                box_rect = Rect(p1, p2)

                if overlap(temp_rect, box_rect):

                    pass

                else:

                    # NOTE: another caution for row and height...
                    cropped = img[x_1:x_2, y_1:y_2, :]

                    # print("NEGATIVE")
                    # print(cropped.shape)
                    # print(img_path)
                    # print(box_path)
                    if int(cropped.shape[0]) == 0 or int(cropped.shape[1]) == 0:
                        break
                        # print("0 Size image error!")
                        # print(img_path)
                        # print(box_path)
                        # raise ValueError("WTF YO")

                    cropped = cv2.resize(cropped, img_size)
                    out_images.append(cropped)
                    # out_labels.append('WTF MATE')
                    out_labels.append(0)
                    break

        return out_images, out_labels

    else:

        for box in boxes:

            # NOTE: keep in mind height and row switching in numpy array!!
            cropped = img[box[1][1]:box[2][1], box[1][0]:box[2][0], :]
            # print(cropped.shape)
            # print(img_path)
            # print(box_path)
            if int(cropped.shape[0]) == 0 or int(cropped.shape[1]) == 0:
                break
                # print("0 Size image error!")
                # print(img_path)
                # print(box_path)
                # raise ValueError("WTF YO")
            cropped = cv2.resize(cropped, img_size)
            out_images.append(cropped)
            # out_labels.append('HAND')
            out_labels.append(1)

        return out_images, out_labels


class DataGen(object):
    """DataGen
    Custom DataGenerator
    """

    def __init__(self, img_size, batch_size, verbose=0, 
                 shuffle=True, negative=False, model=None, bottleneck=False):
        """__init__

        :param img_size:
        :param batch_size:
        :param shuffle:
        :param negative:
        :param model:
        :param bottleneck:
        """

        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.negative = negative
        self.model = model
        self.bottleneck = bottleneck
        self.verbose = verbose

    def generate_train(self, img_list, box_list):

        """generate_batch
        Takes file paths and generate a batch of images and labels to plugin to
        training pipeline.  Augmentation can be also implemented here.

        :param img_list: list of image file path
        :param box_list: list of box file path
        :param img_size: img size for the generator to generate
        :param batch_size: batch size
        """

        while True:

            batch_images = np.empty((self.batch_size, self.img_size[0],
                                     self.img_size[1], 3))
            # batch_labels = np.empty(batch_size, str)
            # Change this later to numpy array with one hot encoding possibly
            batch_labels = []

            batch_full = False

            batch_count = 0

            while not batch_full:

                index = np.random.randint(0, len(img_list))
                img_path = img_list[index]
                box_path = box_list[index]

                if self.negative:

                    if random() < 0.5:

                        out_images, out_labels = crop_images(img_path, box_path,
                                                             img_size=self.img_size)

                    else:

                        out_images, out_labels = crop_images(img_path, box_path,
                                                             img_size=self.img_size,
                                                             negative=True)
                else:

                    out_images, out_labels = crop_images(img_path, box_path,
                                                         img_size=self.img_size)

                # Check if any images returned from cropping
                if len(out_images) < 1:
                    break

                if len(out_images) != len(out_labels):
                    raise ValueError("Images and Lables not matching!")

                for img, label in zip(out_images, out_labels):

                    batch_images[batch_count] = img
                    batch_labels.append(label)
                    batch_count += 1

                    if batch_count == self.batch_size:
                        batch_full = True
                        batch_labels = np_utils.to_categorical(batch_labels, 2)
                        break

            if self.bottleneck and self.model:

                batch_images = preprocess_input(batch_images)
                batch_images = self.model.predict_on_batch(batch_images)

            # print("\nNew Batch Generated with {} images and {} labels!\n".format(len(batch_images), len(batch_labels)))
            if self.verbose:
                print(len(batch_images), batch_images[0].shape)
                print(len(np.unique(batch_labels)), batch_labels[0])
            yield batch_images, batch_labels


def batch_visualize(batch_images, batch_labels):

    """batch_visualize
    Visualize batch of generate images along with labels

    :param batch_images: output from generate_batch function
    :param batch_labels: output from generate_batch function
    """

    plt.figure(figsize=(16, 8))

    for i in range(10):

        plt.subplot(2, 5, i + 1)
        plt.imshow(batch_images[i])
        plt.title(batch_labels[i])

    plt.show()


def plot_model_history(model_history):
    """plot_model_history
    plot accuracy and loss graph vs epoch.

    :param model_history:
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc'])+1),
                model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc'])+1),
                model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc'])+1),
                      len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss'])+1),
                model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1),
                model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss'])+1),
                      len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('./plots/performance.png', bbox_inches='tight')
    # plt.show()


def main():
    """main"""
    # return extract_box(box_list[0])
    pass


if __name__ == "__main__":
    main()
