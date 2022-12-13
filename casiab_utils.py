import os
import numpy as np
import config
from img_utils import get_imgs_from_path
from GEI import create_GEI
from matplotlib import pyplot as plt


def get_training_validation_data():
    train_paths, train_labels, val_paths, val_labels = get_paths_labels()
    train_imgs, val_imgs = get_imgs_from_path(train_paths, val_paths)
    train_data, val_data = create_GEI(train_imgs, val_imgs)

    return train_data, train_labels, val_data, val_labels


def get_paths_labels():
    train_dirs = ["nm-01", "nm-02", "nm-03", "nm-04"]
    val_dirs = ["nm-05", "nm-06"]
    train_paths = np.array((0, 0))
    train_labels = np.array((0, 0))
    val_paths = np.array((0, 0))
    val_labels = np.array((0, 0))

    for i in range(1, 2):
        id = "%03d" % i

        for dir in train_dirs:
            i = 0
            path = "%s/%s/%s/090" % (config.Paths.casia_b_path, id, dir)
            if os.path.exists(path):
                # Update train_labels and train_paths using the correct approach
                train_labels = np.concatenate((train_labels, [id]))
                img_paths = os.listdir(path)

                temp_train_paths = np.zeros((4, 400))
                print(temp_train_paths)
                fuck = np.array(["%s/%s" % (path, img_path) for img_path in img_paths])
                temp_train_paths[i] = fuck
                print(temp_train_paths)
                i += 1
                train_paths = np.concatenate((train_paths, ["%s/%s" % (path, img_path) for img_path in img_paths]))

            else:
                print("path not exist: ", path)

        for dir in val_dirs:
            path = "%s/%s/%s/090" % (config.Paths.casia_b_path, id, dir)
            if os.path.exists(path):
                # Update val_labels and val_paths using the correct approach
                val_labels = np.concatenate((val_labels, [id]))
                img_paths = os.listdir(path)
                img_paths = np.array(img_paths)
                val_paths = np.concatenate((val_paths, ["%s/%s" % (path, img_path) for img_path in img_paths]))

    train_paths = np.delete(train_paths, [0, 1])
    val_paths = np.delete(val_paths, [0, 1])
    return train_paths, train_labels, val_paths, val_labels
    # return np.asarray(train_paths), np.asarray(train_labels), np.asarray(val_paths), np.asarray(val_labels)


if __name__ == "__main__":
    train_data, train_labels, val_data, val_labels = get_training_validation_data()
    # display image in train_data[0] uisng imshow
    plt.imshow(train_data[0])
    plt.show()
