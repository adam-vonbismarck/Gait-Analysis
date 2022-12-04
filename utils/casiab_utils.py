import os
import numpy as np
from .. import config
from img_utils import get_imgs_from_path
from .. GEI import create_GEI

def get_training_validation_data():
    train_paths, train_labels, val_paths, val_labels = get_paths_labels()
    train_imgs, val_imgs = get_imgs_from_path(train_paths, val_paths)
    train_data, val_data = create_GEI(train_imgs, val_imgs)

    return train_data, train_labels, val_data, val_labels


def get_paths_labels():
    train_dirs = ["nm-01", "nm-02", "nm-03", "nm-04"]
    val_dirs = ["nm-05", "nm-06"]
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    for i in range(1, 125):
        id = "%03d" % i

        for dir in train_dirs:
            path = "%s/%s/%s/090" % (config.Paths.casia_b_path, id, dir)
            if os.path.exists(path):
                train_labels.append(id)
                img_paths = os.listdir(path)
                train_paths.append(["%s/%s" % (path, img_path) for img_path in img_paths])
        
        for dir in val_dirs:
            path = "%s/%s/%s/090" % (config.Paths.casia_b_path, id, dir)
            if os.path.exists(path):
                val_labels.append(id)
                img_paths = os.listdir(path)
                val_paths.append(["%s/%s" % (path, img_path) for img_path in img_paths])

    return np.asarray(train_paths), np.asarray(train_labels), np.asarray(val_paths), np.asarray(val_labels)