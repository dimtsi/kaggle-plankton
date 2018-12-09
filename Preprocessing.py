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
import importlib
import NNs
import math
importlib.reload(NNs)
from NNs import *
from NNs import ResNetDynamic, ResNetMine, CNN, SuperNet, EnsembleClassifier

from torchsummary import summary
# %matplotlib inline


def calc_means_stds(image_list):
    np_images = []

    for im in image_list:
        new_im = np.array(im)
        np_images.append(new_im)
    np_images = np.array(np_images)

    img_mean = np.mean(np_images)/255
    img_std = np.std(np_images)/255
    return (img_mean, img_std)

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


import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.labels[idx,0]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples



# In[7]:
#Transforms and Dataset Creation
def create_train_val_datasets(X_train, y_train, X_val = None, y_val = None, norm_params = None):

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
            # transforms.RandomAffine(16),
            transforms.ToTensor(),
            transforms.Normalize(mean=[norm_params['train_norm_mean']],
                        std =[norm_params['train_norm_std']])
        ])

        train_dataset = ListsTrainDataset(X_train, y_train, transform = train_transforms)

        if y_val is not None:
            test_dataset = ListsTrainDataset(X_val, y_val, transform = val_transforms)
        else:
            test_dataset = ListsTestDataset(X_val, transform = test_transforms)

        return (train_dataset, test_dataset)




# In[8]:


def save_model(epoch, model, optimizer, scheduler):
    train_state = {
    'epoch': epoch,
    # 'model' : model,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
    }
    torch.save(train_state, 'trained_model.pt')

# In[9]:

# class_weights = class_weights.type(torch.FloatTensor)
#=============================TRAINING ===================================#

def train_only(model, train_loader, num_epochs):
    learning_rate = 0.001
    weight_decay = 0
    batch_size = train_loader.batch_size
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay);
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.1, patience=10, verbose=True)
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
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay);
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max', factor=0.1, patience=10, verbose=True)
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate);
    #Training
    history = {'batch': [], 'loss': [], 'accuracy': []}
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        tic=timeit.default_timer()
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
            accuracies_train.append(accuracy_train.cpu())
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
        if epoch+1 == 70 and learning_rate == 0.001:
            for group in optimizer.param_groups:
                if 'lr' in group.keys():
                    group['lr'] = 1e-4
                    scheduler._reset()
                    print("MANUAL CHANGE OF LR")
        print(toc-tic)
    return model



# In[ ]:


# from torchvision.models.resnet import *




def train_on_whole():
    train_transforms = transforms. Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_norm_train],
                    std =[mean_std_train])
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

def predict_test_set(model, filenames,  mean_norm_test, std_norm_test):
    test_transforms = transforms. Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_norm_test],
                    std =[std_norm_test])
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
# class_sample_counts = np.bincount(y_train)
# class_sample_counts
# class_weights = 1./torch.Tensor(class_sample_counts)
# train_samples_weight = [class_weights[class_id] for class_id in y_train]



if __name__ == "__main__":
    print("weighted classes")

    train_images = pickle.load(open("pkl/classified_padded64.pkl", "rb"))
    train_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))
    test_images = pickle.load(open("pkl/test_padded64.pkl", "rb"))
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import timeit

    ##Class weights for imbalance
    # from sklearn.utils.class_weight import compute_class_weight
    # labels_df = pd.read_csv('train_onelabel.csv')
    # class_weights = compute_class_weight('balanced', np.arange(121), labels_df['class'])
    # class_weights = np.interp(class_weights, (class_weights.min(), class_weights.max()), (0, +1))
    # class_weights = torch.from_numpy(class_weights).float().to(device)

    from sklearn.model_selection import StratifiedKFold

    pretrained = resnet50(pretrained = True)
    cnn1 = ResNetDynamic(pretrained.block, pretrained.layers,
                num_layers = 2, pretrained_nn = None)

    cnn2 = ResNetDynamic(pretrained.block, pretrained.layers,
                num_layers = 2, pretrained_nn = None)
    #
    cnn1.load_state_dict(torch.load('best_model.pt')['state_dict'])
    cnn2.load_state_dict(torch.load('best2.pt')['state_dict'])

    # cnn2 = ResNetDynamic(Bottleneck, [2, 2, 2, 3],num_layers = 4)
    models = []
    models.append(cnn1)
    models.append(cnn2)
    cnn = EnsembleClassifier(models)



    trained_models = []
    def run_KFolds():
        kf = StratifiedKFold(n_splits=90, random_state=None, shuffle=True)
        for train_indexes, validation_indexes in kf.split(X = train_images, y = train_labels):
            X_train = []
            y_train = []
            X_val = []
            y_val = []
            norm = {}

            for i in train_indexes:
                X_train.append(train_images[i])
                y_train.append(train_labels[i])
            for j in validation_indexes:
                X_val.append(train_images[j])
                y_val.append(train_labels[j])

            print("train: "+ str(len(X_train)) + " val: " +str(len(X_val)))
            norm['train_norm_mean'], norm['train_norm_std'] = calc_means_stds(X_train)

            class_sample_counts = np.bincount(y_train)
            class_sample_counts
            class_weights = 1./torch.Tensor(class_sample_counts)
            train_samples_weight = [class_weights[class_id] for class_id in y_train]

            ## Create Datasets and Dataloaders
            train_dataset, val_dataset = create_train_val_datasets(X_train, y_train, X_val, y_val, norm_params =norm)
            # train_sampler = ImbalancedDatasetSampler(train_dataset)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,
                shuffle = True, num_workers=4)

            test_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size = 32, shuffle = False)

            #Training
            # if torch.cuda.device_count() > 1:
            #   print("Let's use", torch.cuda.device_count(), "GPUs!")
            #   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 cGPUs
            # #   cnn = nn.DataParallel(cnn)
            #   cnn = nn.DataParallel(cnn, device_ids=[0, 1])
            cnn.to(device)

            # cnn = CNN().cuda()
            # summary(cnn, (1,64,64))

        #     print(summary(cnn, (1,28,28)))
            trained_model = train_and_validate(cnn, train_loader, test_loader, num_epochs=100)
            trained_models.append(trained_model)
            break

    run_KFolds()

    final_model = cnn
    final_model.load_state_dict(torch.load('trained_model.pt')['state_dict'])
    mean_norm_test, std_norm_test = calc_means_stds(train_images)
    predict_test_set(final_model, test_filenames, mean_norm_test, std_norm_test)
