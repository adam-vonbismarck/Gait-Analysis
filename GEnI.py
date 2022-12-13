import numpy as np
import config


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

        final = np.zeros((sequence.shape[0], config.Parameters.preprocess_image_height,
                                  config.Parameters.preprocess_image_width))

        for i, person in enumerate(sequence):
            number_images = person.shape[0]
            white_pixel_count_by_pixel = np.sum(np.array(person), axis=0) / 255
            probability_white_pixel = white_pixel_count_by_pixel / number_images
            probability_black_pixel = (number_images - white_pixel_count_by_pixel) / number_images

            entropy_white_component = - (probability_white_pixel * np.log(probability_white_pixel,
                                                                          where=probability_white_pixel > 0))
            entropy_black_component = - (probability_black_pixel * np.log(probability_black_pixel,
                                                                          where=probability_black_pixel > 0))
            total_entropy = entropy_white_component + entropy_black_component
            entropy_min = np.min(total_entropy)
            entropy_max = np.max(total_entropy)
            final[i] = ((total_entropy - entropy_min) * 255) / (entropy_max - entropy_min)

        return final

    train_data = get_GEnI(train_imgs)
    val_data = get_GEnI(val_imgs)

    return train_data, val_data
