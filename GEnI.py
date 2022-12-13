import numpy as np

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

            resultImg = np.sum(np.array(person), axis=0) / 255

            one_pixel_probability = resultImg / person.shape[0]
            zero_pixel_probability = (person.shape[0] - resultImg) / person.shape[0]

            h = -np.nan_to_num(zero_pixel_probability * np.log2(zero_pixel_probability)) - \
                np.nan_to_num(one_pixel_probability * np.log2(one_pixel_probability))

            h_min = np.min(h)
            h_max = np.max(h)
            final[i] = (h - h_min) * 255 / (h_max - h_min)

        return final

    train_data = get_GEnI(train_imgs)
    val_data = get_GEnI(val_imgs)

    return train_data, val_data