"""
This file preprocess the images that are used to train the gait recognition model
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
import config
import cv2


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

    return resize(shifted_image, (config.Parameters.preprocess_image_height, config.Parameters.preprocess_image_width))


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


def preprocess_video(video_filepath):
    video = cv2.VideoCapture(video_filepath)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    scene_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scene_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    i = 0
    was_read = True
    frames = np.zeros((int(total_frames / 3), scene_height, scene_width))

    while (i < (int(total_frames / 3) - 1) and was_read):
        was_read, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[i] = gray_frame
        i += 1

    plt.imshow(frames[0], cmap='gray')
    plt.show()
    return frames


def preprocess(image):
    """
    Preprocesses the images by extracting the human silhouette and then centering the human.
    :param image: The image to preprocess
    :return: The preprocessed images
    """
    extracted_image = extract_human(image)
    preprocessed_image = centre_human(extracted_image)
    return preprocessed_image


if __name__ == '__main__':
    preprocess_video('/Users/adamvonbismarck/Downloads/dataset/IMG_2276.MOV')
