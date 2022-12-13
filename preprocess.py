"""
This file preprocess the images that are used to train the gait recognition model
"""

import os
from re import L
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

PREPROCESSED_WIDTH = 70
PREPROCESSED_HEIGHT = 210


def centre_human(image):
    """Centres human based off of the data from extract_human()"""
    extracted_height, extracted_width = image.shape
    sum_white_pixels = np.sum(image == 255)
    pixel_proportions = [(np.sum(image[:, i] == 255) / sum_white_pixels) for i in range(extracted_width)]
    for i in range(extracted_width - 1):
        pixel_proportions[i + 1] += pixel_proportions[i]
    pixel_proportions = [abs(x - 0.5) for x in pixel_proportions]
    sorted_proportions = np.argsort(pixel_proportions)
    pixel_offset = int((extracted_width / 2) - sorted_proportions[0])
    shifted_image = np.zeros((extracted_height, extracted_width + abs(pixel_offset)))
    if pixel_offset < 0:
        shifted_image[:, :extracted_width] = image
    if pixel_offset >= 0:
        shifted_image[:, abs(pixel_offset):] = image

    return resize(shifted_image, (PREPROCESSED_HEIGHT, PREPROCESSED_WIDTH))


def extract_human(image):
    """
    Get human silhouette from set of images. This method finds the row at which the first white pixel occurs.
    It then finds the row at which the last white pixel occurs. It then crops the image to the range of rows
    between the first and last white pixel. This method assumes that the human silhouette is the only white
    object in the image.
    :param image: The image to extract the human silhouette from
    :return: The image with the human silhouette cropped out
    """
    # find the white pixels
    white_pixels = np.where(image == 255)

    # find the min and max of the white pixels
    min_row = np.min(white_pixels[0])
    max_row = np.max(white_pixels[0])
    min_col = np.min(white_pixels[1])
    max_col = np.max(white_pixels[1])
    # crop the image
    cropped_image = image[min_row:max_row, min_col:max_col]
    return cropped_image


def preprocess(image):
    """
    Preprocesses the images by extracting the human silhouette and then centering the human.
    :param images: The images to preprocess
    :return: The preprocessed images
    """
    extracted_image = extract_human(image)
    preprocessed_image = centre_human(extracted_image)
    return preprocessed_image


'''
def extract_humans_from_folder(folder_path, save_path):
    """
    Extracts the human silhouettes from a folder of images and saves them to a folder
    :param folder_path: The path to the folder of images
    :param save_path: The path to the folder to save the images to
    :return: None
    """
    # get the list of images
    image_list = os.listdir(folder_path)
    # loop through the images
    for i in range(len(image_list)):
        # get the image
        image = imread(folder_path + image_list[i])
        # extract the human
        cropped_image = extract_human(image)
        # save the image
        imsave(save_path + image_list[i], cropped_image)
        # print the progress
        print("Progress: " + str(i) + "/" + str(len(image_list)))
'''

if __name__ == '__main__':
    img = imread('/Users/adamvonbismarck/Downloads/GaitDatasetB-silh/101/nm-01/090/101-nm-01-090-018.png', as_gray=True)
    img = preprocess(img)
