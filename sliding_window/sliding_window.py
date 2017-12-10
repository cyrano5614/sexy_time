import cv2
import numpy as np
from scipy.ndimage.measurements import label


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[2]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[1]
    print('Starting sliding window {} from {}, {} to {}, {}'.format(
        xy_window, x_start_stop[0], y_start_stop[0], x_start_stop[1], y_start_stop[1]))
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    print('Finished with {} windows to slide'.format(len(window_list)))
    return window_list


def search_windows(prediction_method, img, windows, xy_window=(64, 64)):
    print('Starting to search {} windows'.format(len(windows)))
    on_windows = []
    for window in windows:
        crop_img = cv2.resize(img[0][window[0][1]:window[1][1], window[
                              0][0]:window[1][0]], (224, 224))
        crop_img = np.expand_dims(crop_img, axis=0)
        prediction = prediction_method(crop_img)
        if prediction:
            on_windows.append(window)
            print('Found window: {}'.format({window}))

    print('Finished searching windows with {} results'.format(len(on_windows)))
    return on_windows


def heatmap_windows(img, on_windows, threshold):
        # array of zeros with image dimensions
    zeros = np.zeros_like(img[:, :, 0]).astype(np.float)
    # adding heat (+1) per on_window pixel
    heat = add_heat(zeros, on_windows)
    # applying threshold for amount of heat applied to pixels
    heat_thresh = apply_threshold(heat, threshold)
    heatmap = np.clip(heat_thresh, 0, 255)
    labels = label(heatmap)
    
    return draw_labeled_bboxes(img.copy(), labels)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    bbox_list = []
    for label_number in range(1, labels[1]+1):
        nonzero_index = (labels[0] == label_number).nonzero()
        # Identify x and y values of those pixels
        nonzero_y = np.array(nonzero_index[0])
        nonzero_x = np.array(nonzero_index[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzero_x), np.min(nonzero_y)),
                (np.max(nonzero_x), np.max(nonzero_y)))
        bbox_list.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img, bbox_list
