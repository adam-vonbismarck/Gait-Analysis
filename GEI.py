import numpy as np
from preprocess import preprocess


def create_GEI(train_imgs, val_imgs):
    """
    This method creates the GEI for each person
    :param train_imgs: train images for each person
    :param val_imgs: validation images for each person
    :return: GEI for each person
    """
    train_imgs_pre = train_imgs
    val_imgs_pre = val_imgs

    train_data = np.zeros((train_imgs_pre.shape[0]), dtype=object)
    val_data = np.zeros((val_imgs_pre.shape[0]), dtype=object)

    for i in range(train_imgs_pre.shape[0]):
        train_data[i] = np.mean(train_imgs_pre[i], axis=0)

    for i in range(val_imgs_pre.shape[0]):
        val_data[i] = np.mean(val_imgs_pre[i], axis=0)

    return train_data, val_data


def create_GEnI(train_imgs, val_imgs):
    """
    This method creates the GEnI for each person
    :param train_imgs: numpy array of training images for each person
    :param val_imgs: numpy array of validation images for each person
    :return: GEnI for each person
    """

    def get_GEnI(sequence):
        """
        This method creates the GEnI for a sequence of images
        :param sequence: sequence of images
        :return: generated GEnI
        """
        result = np.zeros((sequence.shape[0]), dtype=object)
        for p in range(sequence.shape[0]):
            for i in range(sequence[p].shape[0]):
                # for i in range(frame.shape[0]):
                #     for j in range(frame.shape[1]):
                #         pixel_value = frame[i, j]
                #         pixel_probability = pixel_value / np.sum(frame)
                #         entropy = -1 * pixel_probability * np.log(pixel_probability)
                #         result[i, j] += entropy
                for a in range(sequence[p, i].shape[0]):
                    for b in range(sequence[p, i].shape[1]):
                        pixel_value = sequence[p, i][a, b]
                        pixel_probability = pixel_value / np.sum(sequence[p, i])
                        entropy = -1 * pixel_probability * np.log(pixel_probability)
                        result[a, b] += entropy
                # pixel_probabilities = sequence[p][i] / np.sum(sequence[p][i], axis=None)
                # entropy = -1 * np.sum(pixel_probabilities * np.log(pixel_probabilities), axis=None)
                # result[p] += entropy
            print(result)
            result[p] /= result[p].shape[0]
        return result

    train_data = get_GEnI(train_imgs)
    val_data = get_GEnI(val_imgs)

    return train_data, val_data
