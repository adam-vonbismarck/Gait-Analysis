import numpy as np
from skimage.io import imread

def get_imgs_from_path(train_paths, val_paths):
    train_imgs = np.array(train_paths.shape())
    val_imgs = np.array(val_paths.shape())

    for i in range(train_imgs.size()):
        train_imgs[i] = imread(train_paths[i])

    for i in range(val_imgs.size()):
        val_imgs[i] = imread(val_paths[i])

    return train_imgs, val_imgs