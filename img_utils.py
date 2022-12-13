import numpy as np
from skimage.io import imread

def get_imgs_from_path(train_paths, val_paths):
    train_imgs = np.array(train_paths.shape)
    val_imgs = np.array(val_paths.shape)
    print(len(train_paths))
    train_imgs = np.empty(len(train_paths), dtype=object)
    val_imgs = np.empty(len(val_paths), dtype=object)
    # for i in range(train_imgs.shape[0]):
    #     for j in range(train_imgs.shape[1]):
    #         train_imgs[i, j] = imread(train_paths[i, j], as_grey=True)
    for i in range(len(train_paths)):
        train_imgs[i] = imread(train_paths[i], as_gray=True)

    # for i in range(val_imgs.shape[0]):
    #     for j in range(val_imgs.shape[1]):
    #         val_imgs[i, j] = imread(val_paths[i, j], as_grey=True)

    for i in range(val_imgs.size()):
        val_imgs[i] = imread(val_paths[i], as_gray=True)

    return train_imgs, val_imgs