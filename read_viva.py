"""
Helper functions for the VIVA hand dataset.
"""

import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from random import random


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

    img_path = path + 'pos/'
    box_path = path + 'posGt/'

    img_list = sorted(glob.glob(img_path + '*'))
    box_list = sorted(glob.glob(box_path + '*'))

    return img_list, box_list


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

    if negative == True:

        while len(out_images) < req_numb:

            x_off = 128
            y_off = 128

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

                    cropped = img[y_1:y_2, x_1:x_2, :]
                    cropped = cv2.resize(cropped, img_size)
                    out_images.append(cropped)
                    out_labels.append('WTF MATE')

        return out_images, out_labels

    else:

        for box in boxes:

            # NOTE: keep in mind height and row switching in numpy array!!
            cropped = img[box[1][1]:box[2][1], box[1][0]:box[2][0], :]
            cropped = cv2.resize(cropped, img_size)
            out_images.append(cropped)
            out_labels.append(box[0])

        return out_images, out_labels


def generate_batch(img_list, box_list, img_size, batch_size, negative=False):
    """generate_batch
    Takes file paths and generate a batch of images and labels to plugin to
    training pipeline.  Augmentation can be also implemented here.

    :param img_list: list of image file path
    :param box_list: list of box file path
    :param img_size: img size for the generator to generate
    :param batch_size: batch size
    """

    while True:

        batch_images = np.empty((batch_size, img_size[0],
                                 img_size[1], 3))
        # batch_labels = np.empty(batch_size, str)
        # Change this later to numpy array with one hot encoding possibly
        batch_labels = []

        batch_full = False

        batch_count = 0

        while batch_full == False:

            index = np.random.randint(0, len(img_list))
            img_path = img_list[index]
            box_path = box_list[index]

            if random() < 0.5:

                out_images, out_labels = crop_images(img_path, box_path,
                                                     img_size=img_size)

            else:

                out_images, out_labels = crop_images(img_path, box_path,
                                                     img_size=img_size,
                                                     negative=True)

            for img, label in zip(out_images, out_labels):

                batch_images[batch_count] = img
                # batch_labels[batch_count] = label
                batch_labels.append(label)
                batch_count += 1

                if batch_count == batch_size:
                    batch_full = True
                    break

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


def main():
    """main"""
    # return extract_box(box_list[0])
    pass


if __name__ == "__main__":
    main()
