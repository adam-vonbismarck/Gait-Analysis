import numpy as np
from skimage.io import imread
from preprocess import preprocess


def get_imgs_from_path(train_paths, val_paths):
    """
    This method reads the images from the paths and returns the images
    :param train_paths: paths to the training images
    :param val_paths: paths to the validation images
    :return: array of images for each person and each validation image
    """
    image_array = np.zeros((train_paths.shape[0]), dtype=object)
    val_array = np.zeros((val_paths.shape[0]), dtype=object)

    for i in range(train_paths.shape[0]):

        data_sample = np.zeros((train_paths[i].size, 210, 70))
        # Import images
        for j, file_path in enumerate(train_paths[i]):
            img = imread(file_path, as_gray=True)
            # Treat edge case where the image is just black
            if np.sum(img == 255) == 0:
                continue
            img = preprocess(img)
            data_sample[j] = img

        image_array[i] = data_sample

    for i in range(val_paths.shape[0]):

        data_sample = np.zeros((val_paths[i].size, 210, 70))
        # Import images
        for j, file_path in enumerate(val_paths[i]):
            img = imread(file_path, as_gray=True)
            # Treat edge case where the image is just black
            if np.sum(img == 255) == 0:
                continue
            img = preprocess(img)
            data_sample[j] = img

        val_array[i] = data_sample

    return image_array, val_array
