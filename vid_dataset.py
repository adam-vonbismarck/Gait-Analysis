import os
import numpy as np
import config
from utils.vid_utils import get_imgs_from_vid_path
from GEI import create_GEI
from GEnI import create_GEnI

def get_vid_training_validation_data(useGEnI=False, useSpecial=False):
    """
    This method runs the preprocessing steps and returns the training and validation data
    :return: training data, training labels, validation data, validation labels
    """
    train_paths, train_labels, val_paths, val_labels = get_paths_labels(useSpecial)
    train_imgs, val_imgs = get_imgs_from_vid_path(train_paths, val_paths)

    if useGEnI:
        train_data, val_data = create_GEnI(train_imgs, val_imgs)
    else:
        train_data, val_data = create_GEI(train_imgs, val_imgs)

    return train_data, train_labels, val_data, val_labels


def get_paths_labels(useSpecial):
    """
    This method returns the paths and labels for the training and validation data
    :return: training paths, training labels, validation paths, validation labels
    """
    # TODO check that the paths are correct
    if useSpecial:
        train_vids = ["walk1", "walk2", "walk3", "walk4", "walk5", "walk6"]
        val_vids = ["special1", "special2", "special3", "special4", "special5", "special6"]
    else:
        train_vids = ["walk1", "walk2", "walk3", "walk4"]
        val_vids = ["walk5", "walk6"]
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    for i in range(1, 10):
        id = "%02d" % i
        for vid in train_vids:
            path = "%s/%s/%s.mov" % (config.Parameters.videos_path, id, vid)
            if os.path.exists(path):
                train_labels.append(id)
                train_paths.append(path)

        for vid in val_vids:
            path = "%s/%s/%s.mov" % (config.Parameters.videos_path, id, vid)
            if os.path.exists(path):
                val_labels.append(id)
                val_paths.append(path)

    return np.array(train_paths), np.array(train_labels), np.array(val_paths), np.array(val_labels)
