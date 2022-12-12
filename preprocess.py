"""
This file preprocess the images that are used to train the gait recognition model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imsave


def centre_human(image):
    """Centres human based off of the data from extract_human()"""
    extracted_width, extracted_height = image.shape
    sum_white_pixels = np.sum(image == 255)
    pixel_proportions = [(np.sum(image[:, i] == 255) / sum_white_pixels) for i in range(extracted_width)]
    pass


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
    plt.imshow(cropped_image)
    plt.show()
    return cropped_image

def preprocess(images):
    for image in images:
        image = centre_human(extract_human(image))
    return images
    

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
    pass
