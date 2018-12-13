import numpy as np
import pickle
import math;
import pandas as pd
import glob
import cv2
import mahotas as mt

def haralick(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


train_images = []
train_labels = []

test_images = []
train_filenames = []
test_filenames = []

# labels_df = pd.read_csv('train_onelabel.csv')
# labels_dict = labels_df.set_index('image')['class'].to_dict()
#
# for filename in labels_df['image'].values: ##to keep mapping with classes
#     image = cv2.imread('train_images/'+filename,0).copy()
#     train_images.append(image)
#     train_labels.append(labels_dict[filename])
#     train_filenames.append(filename)
#
#
for filename in glob.iglob('test_images' +'/*'):
    image = cv2.imread(filename,0).copy()
    test_images.append(image)
    test_filenames.append(filename.replace('test_images/', ''))

train_images_PIL = pickle.load(open("pkl/extraclassified_train_images.pkl", "rb"))
# train_labels = pickle.load(open("pkl/extraclassified_train_labels.pkl", "rb"))

# test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes_extraclassified.pkl", "rb"))
# train_images_no_test = [i for j, i in enumerate(train_images) if j not in test_set_mine_indexes]
# train_labels_no_test = [i for j, i in enumerate(train_labels) if j not in test_set_mine_indexes]
# #
# test_mine_images = [i for j, i in enumerate(original_images) if j in test_set_mine_indexes]
# test_mine_labels = [i for j, i in enumerate(original_labels) if j in test_set_mine_indexes]





for image in train_images_PIL:
    train_images.append(np.asarray(image).copy())

len(train_labels)

pickle.dump( train_images, open("pkl/extraclassified_train_images_cv2.pkl", "wb"))
pickle.dump( test_images, open("pkl/test_images_cv2.pkl", "wb"))

train_haralick = []
test_haralick = []

for im in train_images:
    train_haralick.append(haralick(im))
for im in test_images:
    test_haralick.append(haralick(im))

train_haralick = np.array(train_haralick)
test_haralick = np.array(test_haralick)

pickle.dump(train_haralick, open( "features/extraclassified/train_haralick.pkl", "wb"))
pickle.dump(test_haralick, open( "features/extraclassified/test_haralick.pkl", "wb"))

train_moments = []
test_moments = []

for im in train_images:
    train_moments.append(list(cv2.moments(im).values()))
for im in test_images:
    test_moments.append(list(cv2.moments(im).values()))

train_moments = np.array(train_moments)
test_moments = np.array(test_moments)

pickle.dump(train_moments, open("features/extraclassified/train_moments.pkl", "wb"))
pickle.dump(test_moments, open("features/extraclassified/test_moments.pkl", "wb"))

train_sizes = []
test_sizes = []

for im in train_images:
    train_sizes.append(np.array(im.shape))
for im in test_images:
    test_sizes.append(np.array(im.shape))

train_sizes = np.array(train_sizes)
test_sizes = np.array(test_sizes)

pickle.dump(train_sizes, open("features/extraclassified/train_sizes.pkl", "wb"))
pickle.dump(test_sizes, open("features/extraclassified/test_sizes.pkl", "wb"))
