"""
This file preprocess the images that are used to train the gait recognition model
"""

from xml.dom.expatbuilder import FragmentBuilderNS
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


def video_to_frames(video_filepath):
    video = cv2.VideoCapture(video_filepath)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    scene_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scene_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    frames = np.zeros((int(total_frames / 3), scene_height, scene_width))
    i = 0
    was_read = True

    while (i < (total_frames - 1) and was_read):
        was_read, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ((i % 3) == 0):
            frames[int(i/3)] = gray_frame
        i += 1
    return frames

def process_frames(input_frames):
    output_frames = np.zeros((input_frames.shape[0], 1080, 1180))
    for i in range(input_frames.shape[0]):
        new_frame = input_frames[i][:, 570:1750]
        #new_frame = new_frame - background
        #change 130 to -60
        black_frames = new_frame > 130
        white_frames = new_frame <= 130
        new_frame[black_frames] = 0
        new_frame[white_frames] = 255
        output_frames[i] = new_frame
    return remove_artifacts(output_frames)

def remove_artifacts(messy_sillouhettes):
    clean_sillouhettes = np.zeros((messy_sillouhettes.shape[0], 1080, 1180))
    for i in range(messy_sillouhettes.shape[0]):
        erosion_size = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
        clean_sillouhettes[i] = cv2.erode(messy_sillouhettes[i], kernel)

    plt.imshow(clean_sillouhettes[60], cmap='gray')
    plt.show()
    return clean_sillouhettes


    


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
    # background_frames = video_to_frames('IMG_2276.MOV')
    # background_frame = background_frames[68][:, 570:1750]
    frames = video_to_frames('IMG_2346.MOV')
    process_frames(frames)
