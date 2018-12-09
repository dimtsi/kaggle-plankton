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

train_haralick = []
test_haralick = []

for im in train_images:
    train_haralick.append(haralick(image))
for im in test_images:
    test_haralick.append(haralick(image))

train_haralick = np.array(train_haralick)
test_haralick = np.array(test_haralick)

pickle.dump(train_haralick, open( "features/train_haralick.pkl", "wb"))
pickle.dump(test_haralick, open( "features/test_haralick.pkl", "wb"))

train_moments = []
test_moments = []

for im in train_images:
    train_moments.append(list(cv2.moments(im).values()))
for im in test_images:
    test_moments.append(list(cv2.moments(im).values()))

train_moments = np.array(train_moments)
test_moments = np.array(test_moments)

pickle.dump(train_moments, open("features/train_moments.pkl", "wb"))
pickle.dump(test_moments, open("features/test_moments.pkl", "wb"))

train_sizes = []
test_sizes = []

for im in train_images:
    train_sizes.append(np.array(im.shape))
for im in test_images:
    test_sizes.append(np.array(im.shape))

train_sizes = np.array(train_sizes)
test_sizes = np.array(test_sizes)

pickle.dump(train_sizes, open("features/train_sizes.pkl", "wb"))
pickle.dump(test_sizes, open("features/test_sizes.pkl", "wb"))


#==============CLASSIFIED===============#

classified_train_images = []
classified_train_labels = []
classified_train_haralick = []
classified_train_moments = []
classified_train_sizes = []

for x in range(len(train_images)):
    if train_labels[x] == 0:
        continue
    else:
        classified_train_images.append(train_images[x])
        classified_train_labels.append(train_labels[x])
        classified_train_haralick.append(train_haralick[x])
        classified_train_moments.append(train_moments[x])
        classified_train_sizes.append(train_sizes[x])


pickle.dump(classified_train_images, open("pkl/classified_train_images_cv2.pkl", "wb"))
pickle.dump(train_labels, open("pkl/classified_train_labels_cv2.pkl", "wb"))
pickle.dump(train_haralick, open( "features/classified_train_haralick.pkl", "wb"))
pickle.dump(train_moments, open("features/classified_train_moments.pkl", "wb"))
pickle.dump(train_sizes, open("features/classified_train_sizes.pkl", "wb"))
