#!/usr/bin/env python
# coding: utf-8

# # Loading images

# In[1]:


from PIL import Image
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import math;

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision import transforms

from torchsummary import summary
# %matplotlib inline


# In[2]:


train_images = pickle.load(open("pkl/preprocessed_classified_images.pkl", "rb"))
# train_images = train_images[:1000]
train_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))
train_filenames = pickle.load(open("pkl/train_filenames.pkl", "rb"))
test_images = pickle.load(open("pkl/preprocessed_test_images.pkl", "rb"))
test_filenames = pickle.load(open("pkl/test_filenames.pkl", "rb"))


#PIL

widths, heights = [], []
sumx, sumy = 0, 0
for i in train_images:
    sumx += i.size[0]
    widths.append(i.size[0])
    sumy += i.size[1]
    heights.append(i.size[1])


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(widths)
ax2.hist(heights, color = 'orange')
fig.set_size_inches(12, 5)

avg_width = np.mean(widths)
avg_height = np.mean(heights)
print('Average width {} , Average height: {}'.format(avg_width, avg_height))

norm_mean_width = np.mean(widths)
norm_mean_height = np.mean(heights)


# In[4]:


##CONVERT TO NUMPY TO CALCULATE MEAN,STD PER CHANNEL FOR NORMALIZATION
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# np_train = []
# np_test = []
#
# for im in train_images:
#     np_train.append(np.array(im))
#
# for im in test_images:
#     np_test.append(np.array(im))
#
# arr = np.array(np_train) #len,x_pixels,y_pixels, channels
# per_image_mean = np.mean(np_train, axis=(1,2)) #Shape (32,3)
# per_image_std = np.std(np_train, axis=(1,2)) #Shape (32,3)
#
# pop_channel_mean = np.mean(arr, axis=(0, 1, 2))/255
# pop_channel_std = np.std(arr, axis=(0, 1, 2))/255
# # norm_std_array = array([pop_channel_std, pop_channel_std, pop_channel_std])
# pop_channel_mean
# transforms.Normalize(mean=[0.70426004, 0.70426004, 0.70426004],
#             std =[0.43267642, 0.43267642, 0.43267642])
# transforms.Normalize(mean=[0.95558817, 0.95558817, 0.95558817],
#             std =[0.14618639, 0.14618639, 0.14618639])

# In[5]:


class ListsTrainDataset(Dataset):
    def __init__(self, list_of_images, list_of_labels, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
#         super().__init__()
        self.data = list_of_images
        self.labels = np.asarray(list_of_labels).reshape(-1,1)
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.data[index]
        single_image_label = self.labels[index]
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(single_image)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)


# In[6]:


class ListsTestDataset(Dataset):
    def __init__(self, list_of_images, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = list_of_images
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.data[index]
        if self.transform is not None:
            img_as_tensor = self.transform(single_image)
        # Return image ONLY
        return img_as_tensor

    def __len__(self):
        return len(self.data)


# In[7]:


#Transforms and Dataset Creation
def create_datasets_dataloaders(X_train, y_train, X_test= None, y_test = None, batch_size = 32):
    test_transforms = transforms. Compose([
        # transforms.resize(image, (64, 64)),
#         transforms.CenterCrop(64),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.70426004, 0.70426004, 0.70426004],
                    std =[0.43267642, 0.43267642, 0.43267642])
    ])

    train_transforms = transforms. Compose([
#         transforms.CenterCrop(64),
        # transforms.Grayscale(),
        # transforms.resize(image, (64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=360),
        # transforms.RandomAffine(360, shear=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.70426004, 0.70426004, 0.70426004],
                    std =[0.43267642, 0.43267642, 0.43267642])
    ])

    train_dataset = ListsTrainDataset(X_train, y_train, transform = train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=16)

    if y_test is not None:
        test_dataset = ListsTrainDataset(X_test, y_test, transform = test_transforms)
    else:
        test_dataset = ListsTestDataset(X_test, transform = test_transforms)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return (train_loader, test_loader)


# In[8]:


from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


def save_model(epoch, model, optimizer, scheduler):
    train_state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
    }
    torch.save(train_state, 'trained_model.pt')

# In[9]:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import timeit

##Class weights for imbalance
from sklearn.utils.class_weight import compute_class_weight
labels_df = pd.read_csv('train_onelabel.csv')
class_weights = compute_class_weight('balanced', np.arange(121), labels_df['class'])
class_weights = torch.from_numpy(class_weights)
#=============================TRAINING ===================================#

def train_only(model, train_loader, num_epochs):
    learning_rate = 0.001
    weight_decay = 0
    batch_size = train_loader.batch_size
    criterion = nn.CrossEntropyLoss(weight = class_weights);
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay);
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.1, patience=5, verbose=True)
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate);
    #Training
    history = {'batch': [], 'loss': [], 'accuracy': []}
    for epoch in range(num_epochs):
        model.train().cuda()
        losses = [] #losses in epoch per batch
        accuracies_train = [] #accuracies in epoch per batch
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).to(device)
            labels = Variable(labels).squeeze(1).long().to(device)#.cpu()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            _, argmax = torch.max(outputs, 1)
            accuracy_train = (labels == argmax.squeeze()).float().mean()*100
            accuracies_train.append(accuracy_train)
            # Show progress
            # if (i+1) % 32 == 0:
            log = " ".join([
              "Epoch : %d/%d" % (epoch+1, num_epochs),
              "Iter : %d/%d" % (i+1, len(train_loader.dataset)//batch_size)])
            print('\r{}'.format(log), end = " ")
                # history['batch'].append(i)
                # history['loss'].append(loss.item())
                # history['accuracy'].append(accuracy_train.item())
        epoch_log = " ".join([
          "Epoch : %d/%d" % (epoch+1, num_epochs),
          "Training Loss: %.4f" % np.mean(losses),
          "Training Accuracy: %.4f" % np.mean(accuracies_train)])
        print('\r{}'.format(epoch_log))
        print()
    return model


def train_and_validate(model, train_loader, test_loader, num_epochs):
    learning_rate = 0.001
    weight_decay = 0
    batch_size = train_loader.batch_size
    criterion = nn.CrossEntropyLoss(weight = class_weights);
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay);
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max', factor=0.1, patience=5, verbose=True)
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate);
    #Training
    history = {'batch': [], 'loss': [], 'accuracy': []}
    for epoch in range(num_epochs):
        tic=timeit.default_timer()
        model.train().cuda()
        losses = [] #losses in epoch per batch
        accuracies_train = [] #accuracies in epoch per batch
        best_val_accuracy = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).to(device)
            labels = Variable(labels).squeeze(1).long().to(device)#.cpu()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            _, argmax = torch.max(outputs, 1)
            accuracy_train = (labels == argmax.squeeze()).float().mean()*100
            accuracies_train.append(accuracy_train)
            # Show progress
            # if (i+1) % 32 == 0:
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
        for images, labels in test_loader:
            images = Variable(images).to(device)
            labels= labels.squeeze(1)
            outputs = model(images)
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



# In[ ]:

import importlib
import NNs
import math
importlib.reload(NNs)
from NNs import ResNetDynamic, ResNetMine, CNN, SuperNet
from NNs import *

# from torchvision.models.resnet import *

from sklearn.model_selection import StratifiedKFold

pretrained = resnet50(pretrained = True)
cnn = ResNetDynamic(pretrained.block, pretrained.layers,
            num_layers = 2, pretrained_nn = None)


# cnn2 = ResNetDynamic(Bottleneck, [2, 2, 2, 3],num_layers = 4)
# models = []
# models.append(cnn1)
# models.append(cnn2)
# cnn = SuperNet(models)


trained_models = []
def run_KFolds():
    kf = StratifiedKFold(n_splits=7, random_state=None, shuffle=True)
    for train_indexes, validation_indexes in kf.split(X = train_images, y = train_labels):
        X_train = []
        y_train = []
        X_val = []
        y_val = []

        for i in train_indexes:
            X_train.append(train_images[i])
            y_train.append(train_labels[i])
        for j in validation_indexes:
            X_val.append(train_images[j])
            y_val.append(train_labels[j])
        train_loader, test_loader = create_datasets_dataloaders(
            X_train, y_train, X_val, y_val, batch_size = 32)

        #Training
        # cnn1 = PretrainedResnetMine(pretrained.block, pretrained.layers,
        #  pretrained_nn = pretrained)




        # cnn2 = ResNetMine(Bottleneck, [1, 1, 6, 3])
        # models = []
        # models.append(cnn1)
        # models.append(cnn2)

        # cnn = ResNetMine(Bottleneck, [3, 4, 6, 3])
        # cnn = SuperNet(models)
        # if torch.cuda.device_count() > 1:
        #   print("Let's use", torch.cuda.device_count(), "GPUs!")
        #   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 cGPUs
        # #   cnn = nn.DataParallel(cnn)
        #   cnn = nn.DataParallel(cnn, device_ids=[0, 1])
        cnn.to(device)

        # cnn = CNN().cuda()
        summary(cnn, (1,64,64))

    #     print(summary(cnn, (1,28,28)))
        trained_model = train_and_validate(cnn, train_loader, test_loader, num_epochs=100)
        trained_models.append(trained_model)
        break

run_KFolds()


def train_on_whole():
    train_transforms = transforms. Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.70426004, 0.70426004, 0.70426004],
                    std =[0.43267642, 0.43267642, 0.43267642])
    ])
    train_dataset = ListsTrainDataset(train_images, train_labels, transform = train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

    cnn.to(device)
    # cnn.load_state_dict(torch.load('trained_model.pt')['state_dict'])
    summary(cnn, (1,64,64))
    model = train_only(cnn, train_loader, num_epochs=100)
    return model

# train_on_whole()

# predict on testset
final_model = cnn
final_model.load_state_dict(torch.load('trained_model.pt')['state_dict'])
def predict_test_set(model, filenames):
    test_transforms = transforms. Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.70426004, 0.70426004, 0.70426004],
                    std =[0.43267642, 0.43267642, 0.43267642])
    ])

    test_dataset = ListsTestDataset(test_images, transform = test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False)

    model.eval().to(device)
    predictions = []
    for images in test_loader:
        images = Variable(images).cuda()
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        predictions.extend(prediction.cpu().numpy())
    results_df = pd.DataFrame({'image': test_filenames, 'class': predictions}, columns=['image', 'class'])
    results_df.to_csv('results.csv',sep = ',', index = False)

# final_model
predict_test_set(final_model, test_filenames)
