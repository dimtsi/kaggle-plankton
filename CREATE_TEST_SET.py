#!/usr/bin/env python
# coding: utf-8

# ## Create Test SET

# In[1]:


import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np

# train_images = pickle.load(open("pkl/train_padded64.pkl", "rb"))
# train_labels = pickle.load(open("pkl/train_labels.pkl", "rb"))
#
# kf = StratifiedKFold(n_splits=12, random_state=None, shuffle=True)
# for train_indexes, validation_indexes in kf.split(X = train_images, y = train_labels):
#     X_train = []
#     y_train = []
#     X_val = []
#     y_val = []
#     norm = {}
#     num_val_limit = 0.7*len(y_val)
#     for i in train_indexes:
#         X_train.append(train_images[i])
#         y_train.append(train_labels[i])
#     for j in validation_indexes:
#         X_val.append(train_images[j])
#         y_val.append(train_labels[j])
#     break
#
# pickle.dump(validation_indexes, open("pkl/test_set_mine_indexes.pkl", "wb"))
# count_train = np.bincount(train_labels)
# count_val = np.bincount(y_val)
# import pandas as pd
# pd.set_option('display.max_rows', 150)
# df= pd.DataFrame.from_dict({'train':count_train, 'val':count_val})
# print(df)
#
#
# train_images = pickle.load(open("pkl/classified_padded64.pkl", "rb"))
# train_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))
#
# kf = StratifiedKFold(n_splits=12, random_state=None, shuffle=True)
# for train_indexes, validation_indexes in kf.split(X = train_images, y = train_labels):
#     X_train = []
#     y_train = []
#     X_val = []
#     y_val = []
#     norm = {}
#     num_val_limit = 0.7*len(y_val)
#     for i in train_indexes:
#         X_train.append(train_images[i])
#         y_train.append(train_labels[i])
#     for j in validation_indexes:
#         X_val.append(train_images[j])
#         y_val.append(train_labels[j])
#     break
#
# pickle.dump(validation_indexes, open("pkl/test_set_mine_indexes_classified.pkl", "wb"))
# count_train = np.bincount(train_labels)
# count_val = np.bincount(y_val)
# import pandas as pd
# df= pd.DataFrame.from_dict({'train':count_train, 'val':count_val})
# print(df)

train_images = pickle.load(open("pkl/extraclassified_padded64.pkl", "rb"))
train_labels = pickle.load(open("pkl/extraclassified_train_labels.pkl", "rb"))

kf = StratifiedKFold(n_splits=8, random_state=None, shuffle=True)
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

pickle.dump(validation_indexes, open("pkl/test_set_mine_indexes_extraclassified.pkl", "wb"))
count_train = np.bincount(train_labels)
count_val = np.bincount(y_val)
import pandas as pd
df= pd.DataFrame.from_dict({'train':count_train, 'val':count_val})
len(train_images)


# In[ ]:
