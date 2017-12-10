"""
Helper functions for the VIVA hand dataset.
"""

import glob
import numpy as np
import cv2
import matplotlib.image as mpimg


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


def crop_images(img_path, box_path):
    """crop_images
    Takes input image and box coordinates and returns cropped out images

    :param img: original image path we will extract from
    :param boxes: annotation box coordinates from annotation file
    """

    img = mpimg.imread(img_path)
    boxes = extract_box(box_path)

    out_images = []
    out_labels = []

    for box in boxes:

        cropped = img[box[1][0]:box[2][0], box[1][1]:box[2][1], :]
        out_images.append(cropped)
        out_labels.append(box[0])

    return out_images, out_labels


def generate_batch(img_list, box_list, img_size, batch_size):

    while True:

        batch_images = np.empty((batch_size, img_size[0],
                                 img_size[1], img_size[2]))
        batch_labels = np.empty(batch_size)

        batch_full = False

        batch_count = 0

        while batch_full == False:

            index = np.random.randint(0, len(img_list))
            img_path = img_list[index]
            box_path = box_list[index]

            out_images, out_labels = crop_images(img_path, box_path)

            for img, label in zip(out_images, out_labels):

                batch_images[batch_count] = img
                batch_labels[batch_count] = label
                batch_count += 1

                if batch_count == batch_size:
                    batch_full = True
                    break

        yield batch_images, batch_labels




def main():
    """main"""
    # return extract_box(box_list[0])
    pass


if __name__ == "__main__":
    main()
