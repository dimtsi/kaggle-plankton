import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

test_filenames = pickle.load(open("pkl/test_filenames.pkl", "rb"))
predictions = np.zeros(len(test_filenames)).astype('uint8')
results_df = pd.DataFrame({'image': test_filenames, 'class': predictions}, columns=['image', 'class'])
results_df
results_df.to_csv('results.csv',sep = ',', index = False)

classified = False

if classified == False:
    original_images = pickle.load(open("pkl/train_padded64.pkl", "rb"))
    original_labels = pickle.load(open("pkl/train_labels.pkl", "rb"))
    test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes.pkl", "rb"))
else:
    original_images = pickle.load(open("pkl/classified_padded64.pkl", "rb"))
    original_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))
    test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes_classified.pkl", "rb"))

    # train_images = pickle.load(open("pkl/augmented/classified_padded64.pkl", "rb"))
    # train_labels = pickle.load(open("pkl/augmented/classified_all_labels.pkl", "rb"))
train_images = original_images
train_labels = original_labels

test_images = pickle.load(open("pkl/test_padded64.pkl", "rb"))
test_filenames = pickle.load(open("pkl/test_filenames.pkl", "rb"))

##create separate test set
train_images_no_test = [i for j, i in enumerate(train_images) if j not in test_set_mine_indexes]
train_labels_no_test = [i for j, i in enumerate(train_labels) if j not in test_set_mine_indexes]
#
test_mine_images = [i for j, i in enumerate(original_images) if j in test_set_mine_indexes]
test_mine_labels = [i for j, i in enumerate(original_labels) if j in test_set_mine_indexes]

fold = 0
indexes_by_fold = []

kf = StratifiedKFold(n_splits=4, random_state=None, shuffle=True)
for train_indexes, validation_indexes in kf.split(X = train_images_no_test,
                                                  y = train_labels_no_test):
    indexes_by_fold.append(validation_indexes)

indexes_by_fold

training_indexes = indexes_by_fold[fold]
images_for_nn = []
labels_for_nn = []

for index in training_indexes:
    images_for_nn.append(train_images_no_test[index])
    labels_for_nn.append(train_images_no_test[index])

pickle.dump(indexes_by_fold, open("pkl/train_indexes_no_test_stratified.pkl", "wb"))
