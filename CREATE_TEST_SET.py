#!/usr/bin/env python
# coding: utf-8

# ## Create Test SET

# In[1]:


import pickle
from sklearn.model_selection import StratifiedKFold


train_images = pickle.load(open("pkl/train_padded64.pkl", "rb"))
train_labels = pickle.load(open("pkl/train_labels.pkl", "rb"))

kf = StratifiedKFold(n_splits=9, random_state=None, shuffle=True)
for train_indexes, validation_indexes in kf.split(X = train_images, y = train_labels):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    norm = {}
    num_val_limit = 0.7*len(y_val)
    for i in train_indexes:
        X_train.append(train_images[i])
        y_train.append(train_labels[i])
    for j in validation_indexes:
        X_val.append(train_images[j])
        y_val.append(train_labels[j])
    break

pickle.dump(validation_indexes, open("pkl/test_set_mine_indexes.pkl", "wb"))


# In[2]:

# In[3]:


train_images = pickle.load(open("pkl/classified_padded64.pkl", "rb"))
train_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))

kf = StratifiedKFold(n_splits=9, random_state=None, shuffle=True)
for train_indexes, validation_indexes in kf.split(X = train_images, y = train_labels):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    norm = {}
    num_val_limit = 0.7*len(y_val)
    for i in train_indexes:
        X_train.append(train_images[i])
        y_train.append(train_labels[i])
    for j in validation_indexes:
        X_val.append(train_images[j])
        y_val.append(train_labels[j])
    break

pickle.dump(validation_indexes, open("pkl/test_set_mine_indexes_classified.pkl", "wb"))




# In[ ]:
