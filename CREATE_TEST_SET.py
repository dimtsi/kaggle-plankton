import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np

"""
Create test set to be used always separately for evaluation
"""

train_images = pickle.load(open("pkl/classified_padded64.pkl", "rb"))
train_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))

kf = StratifiedKFold(n_splits=8, random_state=None, shuffle=True)
for train_indexes, validation_indexes in kf.split(X = train_images, y = train_labels):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    norm = {}

    for i in train_indexes:
        X_train.append(train_images[i])
        y_train.append(train_labels[i])
    for j in validation_indexes:
        X_val.append(train_images[j])
        y_val.append(train_labels[j])
    break

pickle.dump(validation_indexes, open("pkl/test_set_mine_indexes_extraclassified.pkl", "wb"))
count_train = np.bincount(train_labels)
count_val = np.bincount(y_val)
