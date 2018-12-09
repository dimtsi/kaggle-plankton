
import numpy as np
import math;
import pickle
import pandas as pd
from collections import OrderedDict
import importlib
import time
import timeit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.autograd import Variable
from torchvision import transforms
import NNs
from NNs import *
from NNs import FeatureBoostedCNN
import math
import glob
import cv2

from torchsummary import summary
from Preprocessing import *
from Preprocessing import ListsTrainDataset, ListsTestDataset

# ## LOAD DATA

# In[2]:

train_images = pickle.load(open("pkl/train_resized64.pkl", "rb"))
# train_images = train_images[:1000]
train_labels = pickle.load(open("pkl/train_labels.pkl", "rb"))
train_filenames = pickle.load(open("pkl/train_filenames.pkl", "rb"))
test_images = pickle.load(open("pkl/test_resized64.pkl", "rb"))
test_filenames = pickle.load(open("pkl/test_filenames.pkl", "rb"))


# ## Load handcrafted features

# In[3]:


train_haralick = pickle.load(open("features/train_haralick.pkl", "rb"))
train_moments = pickle.load(open("features/train_moments.pkl", "rb"))
train_sizes = pickle.load(open("features/train_sizes.pkl", "rb"))

test_haralick = pickle.load(open("features/test_haralick.pkl", "rb"))
test_moments = pickle.load(open("features/test_moments.pkl", "rb"))
test_sizes = pickle.load(open("features/test_sizes.pkl", "rb"))

train_handcrafted_features = np.concatenate([train_haralick, train_moments,  train_sizes], axis =1)
test_handcrafted_features = np.concatenate([test_haralick, test_moments,  test_sizes], axis =1)


# ## New Dataset for Features

# In[4]:


class ListsTrainFeatureDataset(Dataset):
    def __init__(self, list_of_images, list_of_labels, list_of_features, transform=None):
#         super().__init__()
        self.data = list_of_images
        self.labels = np.asarray(list_of_labels).reshape(-1,1)
        self.features = np.asarray(list_of_features)
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.data[index]
        single_image_label = self.labels[index]
        single_image_features = self.features[index,:]
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
        self.features = np.asarray(list_of_features)
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.data[index]
        single_image_features = self.features[index,:]
        if self.transform is not None:
            img_as_tensor = self.transform(single_image)
        # Return image ONLY
        return (img_as_tensor, single_image_features)

    def __len__(self):
        return len(self.data)


# In[5]:


def create_train_val_datasets(X_train, y_train, X_val = None, y_val = None,
                              norm_params = None, train_features = None, val_features = None):

        print(norm_params)
        val_transforms = transforms. Compose([
            # transforms.resize(image, (64, 64)),
            # transforms.RandomCrop(64),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[norm_params['train_norm_mean']],
                        std =[norm_params['train_norm_std']])
        ])

        train_transforms = transforms. Compose([
            # transforms.CenterCrop(64),
            transforms.Grayscale(),
            # transforms.resize(image, (64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=360),
            # transforms.RandomAffine(360, shear=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[norm_params['train_norm_mean']],
                        std =[norm_params['train_norm_std']])
        ])

        train_dataset = ListsTrainFeatureDataset(X_train, y_train, train_features, transform = train_transforms)

        if y_val is not None:
            test_dataset = ListsTrainFeatureDataset(X_val, y_val, val_features, transform = val_transforms)
        else:
            test_dataset = ListsTestFeatureDataset(X_val, val_features, transform = val_transforms)

        return (train_dataset, test_dataset)


# In[6]:


def train_and_validate_with_features(model, train_loader, val_loader, num_epochs):
    learning_rate = 0.001
    weight_decay = 0
    batch_size = train_loader.batch_size
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay);
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max', factor=0.1, patience=5, verbose=True)
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate);
    #Training
    history = {'batch': [], 'loss': [], 'accuracy': []}
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        tic=timeit.default_timer()
        model.train().cuda()
        losses = [] #losses in epoch per batch
        accuracies_train = [] #accuracies in epoch per batch
        for i, (images, labels, features) in enumerate(train_loader):
            images = Variable(images).to(device)
            features = Variable(features).float().to(device)
            labels = Variable(labels).squeeze(1).long().to(device)#.cpu()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model((images, features))
            loss = criterion(outputs, labels)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            _, argmax = torch.max(outputs, 1)
            accuracy_train = (labels == argmax.squeeze()).float().mean()*100
            accuracies_train.append(accuracy_train)
            # Show progress
            if (i+1) % 32 == 0:
                log = " ".join([
                  "Epoch : %d/%d" % (epoch+1, num_epochs),
                  "Iter : %d/%d" % (i+1, len(train_loader.dataset)//batch_size)])
                print('\r{}'.format(log), end=" ")
                # history['batch'].append(i)
                # history['loss'].append(loss.item())
                # history['accuracy'].append(accuracy_train.item())
        epoch_log = " ".join([
          "Epoch : %d/%d" % (epoch+1, num_epochs),
          "Training Loss: %.4f" % np.mean(losses),
          "Training Accuracy: %.4f" % np.mean(accuracies_train)])
        print('\r{}'.format(epoch_log))
        ##VALIDATION SCORE AFTER EVERY EPOCH
        model.eval().to(device)
        correct = 0
        total = 0
        for images, labels, features in val_loader:
            images = Variable(images).to(device)
            labels = labels.squeeze(1)
            features = Variable(features).float().to(device)
            outputs = model((images, features))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu().long() == labels).sum()
            val_accuracy = 100*correct.item() / total
        print('VALIDATION SET ACCURACY: %.4f %%' % val_accuracy)
        scheduler.step(correct.item() / total)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            print("saved best model")
            save_model(epoch, model, optimizer, scheduler)
        toc=timeit.default_timer()
        print(toc-tic)
    return model


# In[7]:


from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
norm ={}
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
    break
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
handcrafted_train = scaler.fit_transform(handcrafted_train)
handcrafted_val = scaler.fit_transform(handcrafted_val)


# In[8]:
class FeatureBoostedCNN(nn.Module):

    def __init__(self, network, num_extra_feats=0, num_classes=121):
        super(FeatureBoostedCNN, self).__init__()
        self.convolutional =  nn.Sequential(*list(network.children())[:-2])
        self.cnn_final_size =  64* network.block.expansion * 2**(network.num_layers-1)
        self.flattened_size = self.cnn_final_size + num_extra_feats
        self.fc1 = nn.Sequential(
            # nn.Linear(self.flattened_size, self.flattened_size//2),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.4)
            )
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(self.flattened_size, num_classes)


    def forward(self, x):
        x1 = self.convolutional(x[0])
        x1 = torch.cat((x1, x[1]),1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        return x1

pretrained = resnet50(pretrained = True)
cnn = ResNetDynamic(pretrained.block, pretrained.layers,
            num_layers = 2, pretrained_nn = None)

# cnn.load_state_dict(torch.load('best_new.pt')['state_dict'])
# feature_extractor_cnn = nn.Sequential(*list(cnn.children())[:-2])
# feature_extractor_cnn
num_handcrafted = handcrafted_train.shape[1]
ensemble_nn = FeatureBoostedCNN(cnn, num_handcrafted)


# In[9]:



train_dataset, val_dataset = create_train_val_datasets(X_train, y_train,
                                                       X_val, y_val,
                                                       norm_params = norm,
                                                       train_features = handcrafted_train,
                                                       val_features = handcrafted_val
                                                       )
# train_sampler = ImbalancedDatasetSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,
    shuffle = True, num_workers=4)

val_loader = torch.utils.data.DataLoader(val_dataset,
                            batch_size = 32, shuffle = False)

# In[10]:




train_and_validate_with_features(ensemble_nn, train_loader, val_loader, num_epochs=100)


# In[ ]:


handcrafted_val.shape
