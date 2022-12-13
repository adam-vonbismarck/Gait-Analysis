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
        
        final = np.zeros((sequence.shape[0], 210, 70))
        for i, person in enumerate(sequence):
            result = np.zeros((210, 70))
            for frame in enumerate(person):
                # for i in range(frame.shape[0]):
                #     for j in range(frame.shape[1]):
                #         pixel_value = frame[i, j]
                #         pixel_probability = pixel_value / np.sum(frame)
                #         entropy = -1 * pixel_probability * np.log(pixel_probability)
                #         result[i, j] += entropy
                pixel_probabilities = frame / np.sum(frame)
                entropy = (-1 * pixel_probabilities) * np.log(pixel_probabilities)
                result += entropy
            final[i] = result / person.shape[0]
        return final

        for person in range(sequence.shape[0]):
            for frame in range(person.shape[0]):


    train_imgs_pre = preprocess(train_imgs)
    val_imgs_pre = preprocess(val_imgs)

    train_data = np.array((train_imgs_pre.shape[0], train_imgs_pre.shape[1], train_imgs_pre.shape[2]))
    val_data = np.array((val_imgs_pre.shape[0], val_imgs_pre.shape[1], val_imgs_pre.shape[2]))

    train_data = get_GEnI(train_imgs_pre)
    val_data = get_GEnI(val_imgs_pre)

    return train_data, val_data
