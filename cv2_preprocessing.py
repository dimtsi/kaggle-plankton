import numpy as np
import pickle
import math;
import pandas as pd
import glob
import cv2
import mahotas as mt

def extract_haralick(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


train_images = []
train_labels = []
train_haralick = []

test_images = []
train_filenames = []
test_filenames = []
test_haralick = []

labels_df = pd.read_csv('train_onelabel.csv')
labels_dict = labels_df.set_index('image')['class'].to_dict()

for filename in labels_df['image'].values: ##to keep mapping with classes
    image = cv2.imread('train_images/'+filename,0).copy()
    train_images.append(image)
    train_labels.append(labels_dict[filename])
    train_filenames.append(filename)
    haralick_features = extract_haralick(image)
    train_haralick.append(haralick_features)
for filename in glob.iglob('test_images' +'/*'):
    image = cv2.imread(filename,0).copy()
    test_images.append(image)
    test_filenames.append(filename.replace('test_images/', ''))
    haralick_features = extract_haralick(image)
    test_haralick.append(haralick_features)
