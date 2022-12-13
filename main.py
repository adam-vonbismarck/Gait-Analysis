from casiab_utils import get_training_validation_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def run_GEI():
    train_x, train_y, val_x, val_y = get_training_validation_data()
    #flatten each image in each array
    train_x = np.array([x.flatten() for x in train_x])
    val_x = np.array([x.flatten() for x in val_x])

    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1, max_depth=100, max_features=100)
    model.fit(train_x, train_y)
    p_y = model.predict(val_x)
    return np.mean(p_y == val_y)

def run_GEnI():
    train_x, train_y, val_x, val_y = get_training_validation_data(useGEnI=True)
    #flatten each image in each array
    train_x = np.array([x.flatten() for x in train_x])
    val_x = np.array([x.flatten() for x in val_x])

    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1, max_depth=100, max_features=100)
    model.fit(train_x, train_y)
    p_y = model.predict(val_x)
    return np.mean(p_y == val_y)

def run_model():
    GEI_accuracy = run_GEI()
    GEnI_accuracy = run_GEnI()
    print("GEI accuracy: %.3f" % GEI_accuracy)
    print("GEnI accuracy: %.3f" % GEnI_accuracy)

if __name__ == '__main__':
    run_model()