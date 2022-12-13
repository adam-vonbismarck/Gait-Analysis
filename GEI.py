import numpy as np


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
