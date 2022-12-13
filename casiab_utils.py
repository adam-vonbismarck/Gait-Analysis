import os
import numpy as np
import config
from img_utils import get_imgs_from_path
from GEI import create_GEI
from GEnI import create_GEnI
from matplotlib import pyplot as plt


def get_training_validation_data(useGEnI=False):
    """
    This method runs the preprocessing steps and returns the training and validation data
    :return: training data, training labels, validation data, validation labels
    """
    train_paths, train_labels, val_paths, val_labels = get_paths_labels()
    train_imgs, val_imgs = get_imgs_from_path(train_paths, val_paths)
    
    if useGEnI:
        train_data, val_data = create_GEnI(train_imgs, val_imgs)
    else:
        train_data, val_data = create_GEI(train_imgs, val_imgs)

    return train_data, train_labels, val_data, val_labels


def get_paths_labels():
    """
    This method returns the paths and labels for the training and validation data
    :return: training paths, training labels, validation paths, validation labels
    """
    # TODO check that the paths are correct
    train_dirs = ["nm-01"] #, "nm-02", "nm-03", "nm-04"]
    val_dirs = ["nm-05"] #, "nm-06"]
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    for i in range(1, 125):
        id = "%03d" % i

        for dir in train_dirs:
            path = "%s/%s/%s/090" % (config.Parameters.casia_b_path, id, dir)
            if os.path.exists(path):
                train_labels.append(id)
                img_paths = os.listdir(path)
                train_paths.append(np.array(["%s/%s" % (path, img_path) for img_path in img_paths]))

        for dir in val_dirs:
            path = "%s/%s/%s/090" % (config.Parameters.casia_b_path, id, dir)
            if os.path.exists(path):
                val_labels.append(id)
                img_paths = os.listdir(path)
                val_paths.append(np.array(["%s/%s" % (path, img_path) for img_path in img_paths]))

    train_paths = np.array(train_paths)

    return np.array(train_paths), np.array(train_labels), np.array(val_paths), np.array(val_labels)


if __name__ == "__main__":
    train_data, train_labels, val_data, val_labels = get_training_validation_data()
    # display image in train_data[0] uisng imshow
    
    plt.imshow(train_data[2], cmap='gray')
    plt.show()
