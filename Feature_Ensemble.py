#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math;
import pickle
import pandas as pd
from collections import OrderedDict
import importlib
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision.models.resnet import *
from torch.autograd import Variable
from torchvision import transforms

import NNs
importlib.reload(NNs)
import math
from NNs import ResNetMine, ResNetDynamic, Bottleneck

import glob
import cv2

from torchsummary import summary
from Preprocessing import *
from Preprocessing import ListsTrainDataset, ListsTestDataset


# ## LOAD DATA

# In[3]:


train_images = pickle.load(open("pkl/train_resized64.pkl", "rb"))
# train_images = train_images[:1000]
train_labels = pickle.load(open("pkl/train_labels.pkl", "rb"))
train_filenames = pickle.load(open("pkl/train_filenames.pkl", "rb"))
test_images = pickle.load(open("pkl/test_resized64.pkl", "rb"))
test_filenames = pickle.load(open("pkl/test_filenames.pkl", "rb"))


# ## Load handcrafted features

# In[4]:


train_haralick = pickle.load(open("features/train_haralick.pkl", "rb"))
train_moments = pickle.load(open("features/train_moments.pkl", "rb"))
train_sizes = pickle.load(open("features/train_sizes.pkl", "rb"))

test_haralick = pickle.load(open("features/test_haralick.pkl", "rb"))
test_moments = pickle.load(open("features/test_moments.pkl", "rb"))
test_sizes = pickle.load(open("features/test_sizes.pkl", "rb"))

train_handcrafted_features = np.concatenate([train_haralick, train_moments,  train_sizes], axis =1)
test_handcrafted_features = np.concatenate([test_haralick, test_moments,  test_sizes], axis =1)


# ### New Dataset

# In[5]:


class ListsTrainFeatureDataset(Dataset):
    def __init__(self, list_of_images, list_of_labels, list_of_features, transform=None):
#         super().__init__()
        self.data = list_of_images
        self.labels = np.asarray(list_of_labels).reshape(-1,1)
        self.features = np.asarray(list_of_features).reshape(-1,1)
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.data[index]
        single_image_label = self.labels[index]
        single_image_features = self.features[index]
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(single_image)
        # Return image and the label
        return (img_as_tensor, single_image_label, single_image_features)

    def __len__(self):
        return len(self.data)



class ListsTestFeatureDataset(Dataset):
    def __init__(self, list_of_images, list_of_features, transform=None):
        self.data = list_of_images
        self.features = np.asarray(list_of_features).reshape(-1,1)
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.data[index]
        if self.transform is not None:
            img_as_tensor = self.transform(single_image)
        # Return image ONLY
        return img_as_tensor

    def __len__(self):
        return len(self.data)


# ## CUSTOM NETWORK

# In[6]:


pretrained = resnet50(pretrained = True)
cnn = ResNetDynamic(pretrained.block, pretrained.layers, num_layers = 2, pretrained_nn = None)
#
cnn.load_state_dict(torch.load('best_new.pt')['state_dict'])
feature_extractor_cnn = nn.Sequential(*list(cnn.children())[:-2])
feature_extractor_cnn


# In[7]:


feature_extractor_dict = feature_extractor_cnn.state_dict()
cnn_dict = cnn.state_dict().copy()
pretrained_dict = {k: v for k, v in cnn_dict.items() if k in feature_extractor_dict}
feature_extractor_dict.update(pretrained_dict)
feature_extractor_cnn.load_state_dict(feature_extractor_dict)
feature_extractor_cnn = feature_extractor_cnn.eval().cuda()


# ## Get features from pretrained NN

# In[8]:


def get_cnn_features(model, x):
    features = ...
    mean_norm_test, std_norm_test = calc_means_stds(train_images)

    test_transforms = transforms. Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_norm_test],
                    std =[std_norm_test])
    ])

    cnn_dataset = ListsTestDataset(x, transform = test_transforms)
    cnn_loader = torch.utils.data.DataLoader(cnn_dataset, batch_size = 32, shuffle = False)

    model.eval().to(device)
    predictions = []
    for i, images in enumerate(cnn_loader):
        images = Variable(images, requires_grad=False).cuda()
        outputs = model(images)
        if i == 0:
            features = outputs.detach().cpu()
        else:
            features = torch.cat((features,outputs.detach().cpu()),0)
    return features.numpy()


# In[9]:


from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kf = StratifiedKFold(n_splits=12, random_state=None, shuffle=True)
for train_indexes, validation_indexes in kf.split(X = train_images, y = train_labels):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    norm = {}

    handcrafted_train = []
    handcrafted_val = []

    for i in train_indexes:
        X_train.append(train_images[i])
        y_train.append(train_labels[i])
        handcrafted_train.append(train_handcrafted_features[i])
    for j in validation_indexes:
        X_val.append(train_images[j])
        y_val.append(train_labels[j])
        handcrafted_val.append(train_handcrafted_features[i])
    norm['train_norm_mean'], norm['train_norm_std'] = calc_means_stds(X_train)

    #Training
    # if torch.cuda.device_count() > 1:
    #   print("Let's use", torch.cuda.device_count(), "GPUs!")
    #   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 cGPUs
    # #   cnn = nn.DataParallel(cnn)
    #   cnn = nn.DataParallel(cnn, device_ids=[0, 1])
    feature_extractor_cnn.eval().to(device)

    # cnn = CNN().cuda()
#     summary(cnn, (1,64,64))

#     print(summary(cnn, (1,28,28)))
    cnn_train_features = get_cnn_features(feature_extractor_cnn, X_train)
    cnn_val_features = get_cnn_features(feature_extractor_cnn, X_val)

#     trained_models.append(trained_model)
    break


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features_train = scaler.fit_transform(handcrafted_train)
scaled_features_val = scaler.fit_transform(handcrafted_val)

FINAL_FEATURES_TRAIN = np.concatenate([cnn_train_features, scaled_features_train], axis = 1)
FINAL_FEATURES_VAL = np.concatenate([cnn_val_features, scaled_features_val], axis = 1)


# In[13]:


FINAL_FEATURES_TRAIN.shape
FINAL_FEATURES_VAL.shape


# In[15]:


# %%time

from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
learning_rates = [1, 0.1, 0.01, 0.001, 0.0001]
max_depth=[4, 5, 6, 7, 10, 15]
n_estimators = [5, 10, 50, 100, 500]
n_estimators


# In[16]:


X_train, X_val = FINAL_FEATURES_TRAIN, FINAL_FEATURES_VAL
y_train, y_test = y_train, y_val


# In[ ]:


# ##ADABOOST

# for lr in learning_rates:
#     for depth in max_depth:
#         for est in n_estimators:

#             abc = AdaBoostClassifier(
#                             DecisionTreeClassifier(max_depth=depth),
#                             n_estimators=est,
#                             learning_rate = lr,
#                             algorithm="SAMME")

#             model = abc.fit(X_train, y_train)

#             y_pred_train = model.predict(X_train)
#             y_pred_val = model.predict(X_val)

#             print("lr:" +str(lr) +" est:"+ str(est)+" depth:"+str(depth) )
#             print("Training Accuracy: " +str(accuracy_score(y_train, y_pred_train)))
#             print("Validation Accuracy: " +str(accuracy_score(y_test, y_pred_val)))


# In[ ]:


def CreateBalancedSampleWeights(y_train, largest_class_weight_coef):
    classes = np.unique(y_train)
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key : value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]] * largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]

    return sample_weights
max_weight_coeff = 0.0654138

train_sample_weight = CreateBalancedSampleWeights(y_train, largest_class_weight_coef=max_weight_coeff)

from xgboost import XGBClassifier
start_time = time.time()

for lr in learning_rates:
    model = XGBClassifier(nthread=-1, learning_rate = lr, min_child_weight = 0.001, max_depth=7)
    # model.fit(X_train, y_train, sample_weight=train_sample_weight)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    print("Training Accuracy: " +str(accuracy_score(y_train, y_pred_train)))
    print("Validation Accuracy: " +str(accuracy_score(y_test, y_pred_val)))

    elapsed_time = time.time() - start_time
    print("elapsed time: "+str(elapsed_time))
    print("depth: "+str(lr))
    print(y_pred_val)
