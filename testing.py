"""test file"""
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
import numpy as np
import os
import logging


def shift_left(img, left=10.0, is_grey=True):
    """
    :param numpy.array img: represented by numpy.array
    :param float left: how many pixels to shift to left, this value can be negative that means shift to
                    right {-left} pixels
    :return: numpy.array
    """
    if 0 < abs(left) < 1:
        left = int(left * img.shape[1])
    else:
        left = int(left)

    img_shift_left = np.zeros(img.shape)
    if left >= 0:
        if is_grey:
            img_shift_left = img[:, left:]
        else:
            img_shift_left = img[:, left:, :]
    else:
        if is_grey:
            img_shift_left = img[:, :left]
        else:
            img_shift_left = img[:, :left, :]

    return img_shift_left


def shift_right(img, right=10.0):
    return shift_left(img, -right)


def shift_up(img, up=10.0, is_grey=True):
    """
    :param numpy.array img: represented by numpy.array
    :param float up: how many pixels to shift to up, this value can be negative that means shift to
                    down {-up} pixels
    :return: numpy.array
    """

    if 0 < abs(up) < 1:
        up = int(up * img.shape[0])
    else:
        up = int(up)

    img_shift_up = np.zeros(img.shape)
    if up >= 0:
        if is_grey:
            img_shift_up = img[up:, :]
        else:
            img_shift_up = img[up:, :, :]
    else:
        if is_grey:
            img_shift_up = img[:up, :]
        else:
            img_shift_up = img[:up, :, :]

    return img_shift_up


def shift_down(img, down=10.0):
    return shift_up(img, -down)


def load_image_path_list(path):
    """
    :param path: the test image folder
    :return:
    """
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg") or x.endswith("png")]
    return result


def image_path_list_to_image_pic_list(image_path_list):
    image_pic_list = []
    for image_path in image_path_list:
        im = imread(image_path)
        image_pic_list.append(im)
    return image_pic_list


def extract_human(img):
    """
    :param img: grey type numpy.array image
    :return:
    """

    left_blank = 0
    right_blank = 0

    up_blank = 0
    down_blank = 0

    height = img.shape[0]
    width = img.shape[1]

    for i in range(height):
        if np.sum(img[i, :]) == 0:
            up_blank += 1
        else:
            break

    for i in range(height - 1, -1, -1):
        if np.sum(img[i, :]) == 0:
            down_blank += 1
        else:
            break

    for i in range(width):
        if np.sum(img[:, i]) == 0:
            left_blank += 1
        else:
            break

    for i in range(width - 1, -1, -1):
        if np.sum(img[:, i]) == 0:
            right_blank += 1
        else:
            break

    img = shift_left(img, left_blank)
    img = shift_right(img, right_blank)
    img = shift_up(img, up_blank)
    img = shift_down(img, down_blank)
    return img


if __name__ == '__main__':
    origImage = imread(
        '/Users/adamvonbismarck/Desktop/cs1430/cs1430-final-project-gait-analysis/GaitDatasetA-silh/xch/00_1/xch-00_1-010.png')

    origImage = extract_human(origImage)
    plt.imshow(origImage, cmap='gray')
    plt.show()
    human_extract_center = resize_image(origImage, (210, 70))
    plt.imshow(human_extract_center, cmap='gray')
    plt.show()