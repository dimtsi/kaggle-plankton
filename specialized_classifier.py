#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pickle
import timeit

import torch
import torch.nn as nn
from torchvision import transforms
from Preprocessing import ListsTestDataset
from torch.autograd import Variable

from NNs import ResNetDynamic
from NNs import *

mean_norm_test, std_norm_test = 0.9555881744885841, 0.14618638954043514
pretrained = resnet50(pretrained = True)


# In[2]:


# print("weighted classes")
classified = False
augmented = False
print('classified: '+str(classified))
print('augmented: '+str(augmented))


if classified == False:
    original_images = pickle.load(open("pkl/train_padded64.pkl", "rb"))
    original_labels = pickle.load(open("pkl/train_labels.pkl", "rb"))
    test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes.pkl", "rb"))
else:
    original_images = pickle.load(open("pkl/classified_padded64.pkl", "rb"))
    original_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))
    test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes_classified.pkl", "rb"))

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_layers = 2
models = []
pretrained = resnet50(pretrained = True)
cnn1 = ResNetDynamic(pretrained.block, pretrained.layers,
            num_layers = 2, pretrained_nn = None)
cnn1_dict = torch.load('models/trained_model_fold0_'+str(num_layers)+'layers.pt')['state_dict']
cnn1.load_state_dict(cnn1_dict)
models.append(cnn1)

cnn2 = ResNetDynamic(pretrained.block, pretrained.layers,
            num_layers = 2, pretrained_nn = None)
cnn2_dict = torch.load('models/trained_model_fold1_'+str(num_layers)+'layers.pt',
                       map_location={'cuda:1': 'cuda:0'})['state_dict']
cnn2.load_state_dict(cnn2_dict)
models.append(cnn2)

cnn3 = ResNetDynamic(pretrained.block, pretrained.layers,
            num_layers = 2, pretrained_nn = None)
cnn3_dict = torch.load('models/trained_model_fold2_'+str(num_layers)+'layers.pt'
                       , map_location={'cuda:2': 'cuda:0'})['state_dict']
cnn3.load_state_dict(cnn3_dict)
models.append(cnn3)

cnn4 = ResNetDynamic(pretrained.block, pretrained.layers,
            num_layers = 2, pretrained_nn = None)
cnn4_dict = torch.load('models/trained_model_fold3_'+str(num_layers)+'layers.pt',
                        map_location={'cuda:3': 'cuda:0'})['state_dict']
cnn4.load_state_dict(cnn4_dict)
models.append(cnn4)


# In[3]:


def predict_cnn(model, images,  mean_norm_test, std_norm_test, multiGPU = False):
    test_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_norm_test],
                    std =[std_norm_test])
    ])

    test_dataset = ListsTestDataset(images, transform = test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False)

    model.eval()
    predictions = []
    total_outputs = []
    for images in test_loader:
        images = Variable(images).cuda()
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        predictions.extend(prediction.cpu().numpy())
        total_outputs.extend((torch.nn.Softmax()(outputs)).detach().cpu().numpy())

    return (total_outputs, predictions) ##Returns Probabilities and predictions


# In[4]:


all_model_train_outputs = []
all_model_train_predictions = []

for model in models:
    train_outputs, train_predictions = predict_cnn(model.cuda(), train_images_no_test, mean_norm_test, std_norm_test)
    cnn_train_outputs.append(train_outputs)
    cnn_train_predictions.append(train_predictions)
    
    test_outputs, test_predictions = predict_cnn(model.cuda(), test_mine_images, mean_norm_test, std_norm_test)
    cnn_test_outputs.append(test_outputs)
    cnn_test_predictions.append(test_predictions)


# In[42]:


np_cnn_train_outputs = np.concatenate(cnn_train_outputs, axis = 1)
np_cnn_test_outputs = np.concatenate(cnn_test_outputs, axis = 1)


# In[43]:


X_train = np_cnn_train_outputs
Y_train = np.array(train_labels_no_test)

X_val = np_cnn_test_outputs 
Y_val = np.array(test_mine_labels) 


# In[ ]:


from xgboost import XGBClassifier
start_time = time.time()
learning_rates = [0.1, 0.01, 0.01, 0.001]
for lr in learning_rates:
    model = XGBClassifier(nthread=-1, learning_rate = lr, min_child_weight = 0.001, max_depth=2)
    # model.fit(X_train, y_train, sample_weight=train_sample_weight)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    print("Training Accuracy: " +str(accuracy_score(y_train, y_pred_train)))
    print("Validation Accuracy: " +str(accuracy_score(y_test, y_pred_val)))

    elapsed_time = time.time() - start_time
    print("elapsed time: "+str(elapsed_time))
    print("lr: "+str(lr))
    print(y_pred_val)

