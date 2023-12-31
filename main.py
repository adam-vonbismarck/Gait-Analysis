from casiab import get_casia_training_validation_data
from vid_dataset import get_vid_training_validation_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def run_GEI(train_x, train_y, val_x, val_y, verbose):
    """
    This method runs the GEI model and returns the accuracy
    :return: the accuracy of the GEI model
    """
    print("Running GEI model")
    # flatten each image in each array
    train_x = np.array([x.flatten() for x in train_x])
    val_x = np.array([x.flatten() for x in val_x])

    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2016, verbose=int(verbose), max_depth=100,
                                   max_features=100)
    model.fit(train_x, train_y)
    p_y = model.predict(val_x)
    if verbose:
        for i, p in enumerate(p_y):
            print("Predicted: %s, Actual: %s" % (p, val_y[i]))
    return np.mean(p_y == val_y)


def run_GEnI(train_x, train_y, val_x, val_y, verbose):
    """
    This method runs the GEnI model and returns the accuracy of the model
    :return: the accuracy of the GEnI model
    """
    print("Running GEnI model")
    # flatten each image in each array
    train_x = np.array([x.flatten() for x in train_x])
    val_x = np.array([x.flatten() for x in val_x])

    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2016, verbose=int(verbose), max_depth=100,
                                   max_features=100)
    model.fit(train_x, train_y)
    p_y = model.predict(val_x)

    if verbose:
        for i, p in enumerate(p_y):
            print("Predicted: %s, Actual: %s" % (p, val_y[i]))
    return np.mean(p_y == val_y)


def run_model(useCasiaB=True, useSpecial=False, verbose=False):
    """
    This method runs the GEI and GEnI models and prints the accuracy of each model
    """
    if useCasiaB:
        train_x, train_y, val_x, val_y = get_casia_training_validation_data()
        GEI_accuracy = run_GEI(train_x, train_y, val_x, val_y, verbose)
        train_x, train_y, val_x, val_y = get_casia_training_validation_data(useGEnI=True)
        GEnI_accuracy = run_GEnI(train_x, train_y, val_x, val_y, verbose)
    else:
        train_x, train_y, val_x, val_y = get_vid_training_validation_data(useSpecial=useSpecial)
        GEI_accuracy = run_GEI(train_x, train_y, val_x, val_y, verbose)
        train_x, train_y, val_x, val_y = get_vid_training_validation_data(useGEnI=True, useSpecial=useSpecial)
        GEnI_accuracy = run_GEnI(train_x, train_y, val_x, val_y, verbose)
    
    print("GEI accuracy: %.3f" % GEI_accuracy)
    print("GEnI accuracy: %.3f" % GEnI_accuracy)


if __name__ == '__main__':
    run_model()
