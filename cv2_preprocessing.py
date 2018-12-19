import numpy as np
import pickle
import math;
import pandas as pd
import glob
import cv2
import mahotas as mt


"""Extract haralick features, image texture moments and sizes to be
used as external handcrafted features"""

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

for filename in glob.iglob('test_images' +'/*'):
    image = cv2.imread(filename,0).copy()
    test_images.append(image)
    test_filenames.append(filename.replace('test_images/', ''))

train_images_PIL = pickle.load(open("pkl/classified_train_images.pkl", "rb"))

for image in train_images_PIL:
    train_images.append(np.asarray(image).copy())

len(train_labels)

pickle.dump( train_images, open("pkl/classified_train_images_cv2.pkl", "wb"))
pickle.dump( test_images, open("pkl/test_images_cv2.pkl", "wb"))

train_haralick = []
test_haralick = []

for im in train_images:
    train_haralick.append(haralick(im))
for im in test_images:
    test_haralick.append(haralick(im))

train_haralick = np.array(train_haralick)
test_haralick = np.array(test_haralick)

pickle.dump(train_haralick, open( "features/classified/train_haralick.pkl", "wb"))
pickle.dump(test_haralick, open( "features/classified/test_haralick.pkl", "wb"))

train_moments = []
test_moments = []

for im in train_images:
    train_moments.append(list(cv2.moments(im).values()))
for im in test_images:
    test_moments.append(list(cv2.moments(im).values()))

train_moments = np.array(train_moments)
test_moments = np.array(test_moments)

pickle.dump(train_moments, open("features/classified/train_moments.pkl", "wb"))
pickle.dump(test_moments, open("features/classified/test_moments.pkl", "wb"))

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
