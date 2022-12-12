import numpy as np
from preprocess import preprocess

def create_GEI(train_imgs, val_imgs):
    #TODO: prepocess images to get centered figures
    train_imgs_pre = preprocess(train_imgs)
    val_imgs_pre = preprocess(val_imgs)

    train_data = np.array((train_imgs_pre.shape[0], train_imgs_pre.shape[1], train_imgs_pre.shape[2]))
    val_data = np.array((val_imgs_pre.shape[0], val_imgs_pre.shape[1], val_imgs_pre.shape[2]))

    for i in train_imgs_pre.shape[0]:
        train_data[i] = np.mean(train_imgs_pre[i], axis=0)

    for i in val_imgs_pre.shape[0]:
        val_data[i] = np.mean(val_imgs_pre[i], axis=0)

    return train_data, val_data
