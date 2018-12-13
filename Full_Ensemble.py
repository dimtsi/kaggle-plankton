#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math;
import pickle
import pandas as pd
from collections import OrderedDict
import importlib
import time
import timeit

import torch
import torch.nn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision.models.resnet import *
from torch.autograd import Variable
from torchvision import transforms

import NNs
from NNs import *
importlib.reload(NNs)
import math
from NNs import ResNetDynamic, FeatureBoostedCNN

import glob
import cv2

from torchsummary import summary
from Preprocessing import *
from Preprocessing import ListsTrainDataset, ListsTestDataset

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[2]:


import sys
print(sys.path)


# ## LOAD DATA

# In[3]:


original_train_images = pickle.load(open("pkl/train_padded64.pkl", "rb"))
# train_images = train_images[:1000]
original_labels = pickle.load(open("pkl/train_labels.pkl", "rb"))
kaggle_test_images = pickle.load(open("pkl/test_padded64.pkl", "rb"))
kaggle_test_filenames = pickle.load(open("pkl/test_filenames.pkl", "rb"))


# ## Load handcrafted features

# In[4]:


original_haralick = pickle.load(open("features/train_haralick.pkl", "rb"))
original_moments = pickle.load(open("features/train_moments.pkl", "rb"))
original_sizes = pickle.load(open("features/train_sizes.pkl", "rb"))

kaggle_test_haralick = pickle.load(open("features/test_haralick.pkl", "rb"))
kaggle_test_moments = pickle.load(open("features/test_moments.pkl", "rb"))
kaggle_test_sizes = pickle.load(open("features/test_sizes.pkl", "rb"))

train_handcrafted_features = np.concatenate([original_haralick, original_moments,  original_sizes], axis =1)
kaggle_test_handcrafted_features = np.concatenate([kaggle_test_haralick, kaggle_test_moments,  kaggle_test_sizes], axis =1)


# ## Split to train test mine

# In[5]:


test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes.pkl", "rb"))

train_images = [i for j, i in enumerate(original_train_images) if j not in test_set_mine_indexes]
train_labels = [i for j, i in enumerate(original_labels) if j not in test_set_mine_indexes]
train_handcrafted = [i for j, i in enumerate(train_handcrafted_features) if j not in test_set_mine_indexes]

#
test_mine_images = [i for j, i in enumerate(original_train_images) if j in test_set_mine_indexes]
test_mine_labels = [i for j, i in enumerate(original_labels) if j in test_set_mine_indexes]
test_mine_handcrafted = [i for j, i in enumerate(train_handcrafted_features) if j in test_set_mine_indexes]


# In[6]:


X_train_cnn, y_train_cnn = train_images, train_labels
X_val_cnn, y_val_cnn = test_mine_images, test_mine_labels


# ## CNN

# In[7]:


pretrained = resnet50(pretrained = True)
cnn = ResNetDynamic(pretrained.block, pretrained.layers, num_layers = 2, pretrained_nn = None)
#
cnn_dict = torch.load('models/all_elements_trained_model_90_new.pt', map_location={"cuda:1": "cuda:0", "cuda:2": "cuda:0"})['state_dict']
cnn.load_state_dict(cnn_dict)
cnn = cnn.eval().cuda()
del(pretrained)
feature_extractor_cnn = nn.Sequential(*list(cnn.children())[:-2]).eval().cuda()


# In[8]:


mean_norm_test, std_norm_test = calc_means_stds(train_images)

def get_cnn_features(feature_extractor, model, x):
    features = ...
    mean_norm_test, std_norm_test = calc_means_stds(train_images)

    test_transforms = transforms. Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_norm_test],
                    std =[std_norm_test])
    ])

    total_features = torch.Tensor().float().cpu()
    total_predicted = torch.Tensor().long()
    total_probabilities = torch.Tensor().float()

    cnn_dataset = ListsTestDataset(x, transform = test_transforms)
    cnn_loader = torch.utils.data.DataLoader(cnn_dataset, batch_size = 32, shuffle = False)

    predictions = []
    for i, images in enumerate(cnn_loader):
        images = Variable(images, requires_grad=False).cuda()
        outputs = model(images).cuda()
        _, predicted = torch.max(outputs.data, 1)
        features = feature_extractor(images)

        total_features = torch.cat((total_features, features.detach().cpu()))
        total_predicted = torch.cat((total_predicted, predicted.cpu().long()))
        total_probabilities = torch.cat((total_probabilities,(torch.nn.Softmax()(outputs)).detach().cpu()))

    return (total_features.numpy(), total_predicted.numpy(), total_probabilities.numpy())


# In[ ]:


cnn_train_features, cnn_train_predictions, cnn_train_probabilities = get_cnn_features(feature_extractor_cnn, cnn, X_train_cnn)
cnn_val_features, cnn_val_predictions, cnn_val_probabilities = get_cnn_features(feature_extractor_cnn, cnn, X_val_cnn)
cnn_kaggle_features, cnn_kaggle_predictions, cnn_kaggle_probabilities = get_cnn_features(feature_extractor_cnn, cnn, kaggle_test_images)


# ## Scale and Preprocess for Ensemble

# In[ ]:


scaler = StandardScaler()
scaled_handcrafted_train = scaler.fit_transform(train_handcrafted)
scaled_handcrafted_val = scaler.fit_transform(test_mine_handcrafted)
scaled_handcrafted_kaggle = scaler.fit_transform(kaggle_test_handcrafted_features)
scaled_cnn_train_features = scaler.fit_transform(cnn_train_features)
scaled_cnn_val_features = scaler.fit_transform(cnn_val_features)
scaled_cnn_kaggle_features = scaler.fit_transform(cnn_kaggle_features)


# ### Setup Features DF

# In[ ]:


feature_names = []

for i in range(original_haralick.shape[1]):
    feature_names.append("haralick"+str(i))
for i in range(original_moments.shape[1]):
    feature_names.append("moments"+str(i))
for i in range(original_sizes.shape[1]):
    feature_names.append("sizes"+str(i))
for i in range(scaled_cnn_train_features.shape[1]):
    feature_names.append("deep"+str(i))


# In[ ]:


##concat handcrafted and deep features

NP_FEATURES_TRAIN = np.concatenate([scaled_handcrafted_train, scaled_cnn_train_features], axis = 1)
NP_FEATURES_VAL = np.concatenate([scaled_handcrafted_val, scaled_cnn_val_features], axis = 1)
NP_FEATURES_KAGGLE = np.concatenate([scaled_handcrafted_kaggle, scaled_cnn_kaggle_features], axis = 1)

y_train = train_labels
y_test = test_mine_labels

X_train = pd.DataFrame(NP_FEATURES_TRAIN, columns = feature_names)
X_test = pd.DataFrame(NP_FEATURES_VAL, columns = feature_names)
X_kaggle = pd.DataFrame(NP_FEATURES_KAGGLE, columns = feature_names)


# ## PCA

# In[ ]:


pca = PCA(n_components=40)

concatenated = np.concatenate([X_train, X_test], axis =0)
concatenated = np.concatenate([concatenated, X_kaggle], axis =0)

principalComponents = pca.fit_transform(concatenated)
principalComponents.shape

x_train = principalComponents[:len(X_train)]
x_test = principalComponents[len(X_train):len(X_train)+len(X_test)]
x_kaggle = principalComponents[len(X_train)+len(X_test):]


# In[ ]:


## Base Learners
import sklearn
import xgboost as xgb
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# In[ ]:


ntrain = x_train.shape[0]
ntest = x_test.shape[0]
nkaggle = x_kaggle.shape[0]

SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(n_splits = NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier




# In[ ]:


def get_results(model, train_data, test_data, training_labels, test_labels):
    y_pred_train = model.predict(train_data)
    y_pred_test = model.predict(test_data)
    print("Training Accuracy: " +str(accuracy_score(training_labels, y_pred_train)))
    print("Validation Accuracy: " +str(accuracy_score(test_labels, y_pred_test)))


# In[ ]:


####Weak Classifiers Params


rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'max_features' : 0.2,
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.2
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# In[61]:


##DTree

# start_time = time.time()
#
# dt_model = DecisionTreeClassifier(random_state=1)
#
# dt_model.fit(x_train, y_train)
# get_results(dt_model, x_train, x_test, y_train, y_test)
#
#
# elapsed_time = time.time() - start_time
# print("elapsed time: "+str(elapsed_time))
#
#
# # In[62]:
#
#
# ##RandomForest
#
# start_time = time.time()
#
# rf_model = RandomForestClassifier(**rf_params)
# rf_model.fit(x_train, y_train)
# get_results(rf_model, x_train, x_test, y_train, y_test)
#
# elapsed_time = time.time() - start_time
# print("elapsed time: "+str(elapsed_time))
#
#
# # In[63]:
#
#
# ##XGBoost
#
# start_time = time.time()
#
# xgb_model = XGBClassifier(nthread=-1, learning_rate = 0.01, min_child_weight = 0.01, max_depth=5, )
# xgb_model.fit(x_train, y_train)
# get_results(xgb_model, x_train, x_test, y_train, y_test)
#
# elapsed_time = time.time() - start_time
# print("elapsed time: "+str(elapsed_time))
#
#
# # In[64]:
lr = [0.15, 0.1, 0.05, 0.01, 0.01]
est = [200,500, 100]
##AdaBoost
for learn in lr:
    for estim in est:
        ada_params['n_estimators']=estim
        ada_params['learning_rate']=learn


        start_time = time.time()

        ada_model = AdaBoostClassifier(**ada_params)
        ada_model.fit(x_train, y_train)
        get_results(ada_model, x_train, x_test, y_train, y_test)

        elapsed_time = time.time() - start_time
        print("elapsed time: "+str(elapsed_time))


# In[24]:


# ##ExtraTrees
#
# start_time = time.time()
#
# et_model = ExtraTreesClassifier(**et_params)
# et_model.fit(x_train, y_train)
# get_results(et_model, x_train, x_test, y_train, y_test)
#
# elapsed_time = time.time() - start_time
# print("elapsed time: "+str(elapsed_time))
#
#
# # In[25]:
#
#
# ##SVM
#
# start_time = time.time()
#
# svm = SVC(**svc_params)
# svm.fit(scaler.fit_transform(x_train), y_train)
# get_results(svm, scaler.fit_transform(x_train), scaler.fit_transform(x_test), y_train, y_test)
#
# elapsed_time = time.time() - start_time
# print("elapsed time: "+str(elapsed_time))
#
#
# # In[26]:
#
#
# from sklearn.ensemble import VotingClassifier
# eclf = VotingClassifier(estimators=[('dt', dt_model), ('et_model', clf2), ('xgb', XGBClassifier), ('ada', XGBClassifier)], voting='soft')
#
#
# # In[27]:
#
#
# svm_results = svm.predict(scaler.fit_transform(x_kaggle))


# In[28]:


# best_results = pd.read_csv('best_results.csv')

# best_results['predicted']=svm_results

# best_results = best_results.drop(columns=['class'])

# final = best_results.rename(index=str, columns={"predicted": "class"})
# final
# final.to_csv('results.csv',sep = ',', index = False)
