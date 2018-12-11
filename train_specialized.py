

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

        if X_val is None and y_val is None:
            return train_dataset

        elif X_val is not None:
            test_dataset = ListsTrainDataset(X_val, y_val, transform = val_transforms)
        else:
            test_dataset = ListsTestDataset(X_val, transform = test_transforms)

        return (train_dataset, test_dataset)




# In[8]:


def save_model(epoch, model, optimizer, scheduler, name = 'trained_model.pt'):
    train_state = {
    'epoch': epoch,
    # 'model' : model,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
    }
    print("Saved model at: "+str(name))
    torch.save(train_state, name)

# In[9]:

# class_weights = class_weights.type(torch.FloatTensor)
#=============================TRAINING ===================================#

def train_and_validate(model, train_loader, test_loader,
                       num_epochs, device = torch.device("cuda:3"),
                       learning_rate = 0.001,
                       weight_decay = 0,
                       multiGPU = False,
                       save_name = 'trained_model.pt'):
    batch_size = train_loader.batch_size
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay);
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate,
    #                                 weight_decay = weight_decay,
    #                                 momentum = 0.6);

    patience = 15 if weight_decay > 0 else 7
    step_size = 25 if weight_decay > 0 else 15

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, 'max', factor=0.1, patience=patience, verbose=True)
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate);
    #Training
    print("lr:{} wd:{}".format(learning_rate, weight_decay))
    model.train().to(device)
    if isinstance(model, EnsembleClassifier):
        if multiGPU == True:
            print("multiGPU")
            model.set_devices_multiGPU()

    history = {'batch': [], 'loss': [], 'accuracy': []}
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        tic=timeit.default_timer()
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

        epoch_log = " ".join([
          "Epoch : %d/%d" % (epoch+1, num_epochs),
          "Training Loss: %.4f" % np.mean(losses),
          "Training Accuracy: %.4f" % np.mean(accuracies_train)])
        print('\r{}'.format(epoch_log))

        ##VALIDATION SCORE AFTER EVERY EPOCH
        model.eval()
        correct = 0
        total = 0
        total_labels = torch.Tensor().long()
        total_predicted = torch.Tensor().long()


        for images, labels in test_loader:
            images = Variable(images).to(device)
            labels= labels.squeeze(1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu().long() == labels).sum()

            total_labels = torch.cat((total_labels,labels))
            total_predicted = torch.cat((total_predicted, predicted.cpu().long()))

            val_accuracy = 100*correct.item() / total
        print('VALIDATION SET ACCURACY: %.4f %%' % val_accuracy)
        # scheduler.step(correct.item() / total)

        ###Results for analysis###
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(epoch, model, optimizer, scheduler, name = save_name)
            pickle.dump(total_predicted.cpu().long(), open("predicted.pkl", "wb"))
            pickle.dump(total_labels.long(), open("training_labels.pkl", "wb"))

        toc=timeit.default_timer()
        if epoch+1 == 70:
            for group in optimizer.param_groups:
                if 'lr' in group.keys():
                    if group['lr'] == 0.001:
                        group['lr'] == 0.0001
                        scheduler._reset()
                        print("MANUAL CHANGE OF LR")
        print(toc-tic)
    return model


# predict on testset

def predict_test_set_kaggle(model, filenames,  mean_norm_test, std_norm_test):
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

        # train_images = pickle.load(open("pkl/augmented/classified_padded64.pkl", "rb"))
        # train_labels = pickle.load(open("pkl/augmented/classified_all_labels.pkl", "rb"))
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

    ###========================MAIN EXECUTION=========================###

    #####Specialization####
    fold = 2
    print('Fold: '+str(fold))

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

    device = torch.device("cuda:"+str(fold) if torch.cuda.device_count()>2 else "cuda:0")
    import timeit

    ##Class weights for imbalance
    # from sklearn.utils.class_weight import compute_class_weight
    # labels_df = pd.read_csv('train_onelabel.csv')
    # class_weights = compute_class_weight('balanced', np.arange(121), labels_df['class'])
    # class_weights = np.interp(class_weights, (class_weights.min(), class_weights.max()), (0, +1))
    # class_weights = torch.from_numpy(class_weights).float().to(device)

    num_layers = 4
    pretrained = resnet50(pretrained = True)
    cnn = ResNetDynamic(pretrained.block, pretrained.layers,
                num_layers = num_layers, pretrained_nn = None)

    from sklearn.model_selection import KFold, StratifiedKFold

    ####==========SPECIALIZED TRAINING========####
    indexes_by_fold = []

    kf = KFold(n_splits=4, random_state=None, shuffle=False) ##no_shuffle_for_class_specialization
    for train_indexes, validation_indexes in kf.split(X = train_images_no_test,
                                                      y = train_labels_no_test):
        indexes_by_fold.append(validation_indexes)

    training_indexes = indexes_by_fold[fold]
    images_for_nn = []
    labels_for_nn = []


    for index in training_indexes:
        images_for_nn.append(train_images_no_test[index])
        labels_for_nn.append(train_labels_no_test[index])

    ####TRAIN####
    trained_models = []
    def run_KFolds():
        num_splits = 15
        print(num_splits)
        kf = StratifiedKFold(n_splits=num_splits, random_state=None, shuffle=True)
        for train_indexes, validation_indexes in kf.split(X = images_for_nn,
                                                          y = labels_for_nn):
            X_train = []
            y_train = []
            X_val = []
            y_val = []
            norm = {}
            for i in train_indexes:
                X_train.append(images_for_nn[i])
                y_train.append(labels_for_nn[i])
            for j in validation_indexes:
                X_val.append(images_for_nn[j])
                y_val.append(labels_for_nn[j])
            # X_train = X_train[:10]
            # y_train = y_train[:10]

            # X_val = test_mine_images
            # y_val = test_mine_labels
            print("train: "+ str(len(X_train)) + " val: " + str(len(X_val))+ " None: " + str(len(original_images)-len(X_train)))
            print(np.bincount(y_train))
            print(np.bincount(y_val))
            norm['train_norm_mean'], norm['train_norm_std'] = calc_means_stds(original_images)

            class_sample_counts = np.bincount(y_train)
            class_sample_counts
            class_weights = 1./torch.Tensor(class_sample_counts)
            train_samples_weight = [class_weights[class_id] for class_id in y_train]

            ## Create Datasets and Dataloaders
            train_dataset, val_dataset = create_train_val_datasets(X_train, y_train,
                                                                   X_val, y_val,
                                                                   norm_params =norm)
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
            trained_model = train_and_validate(cnn, train_loader, test_loader,
                                               num_epochs=200,
                                               learning_rate = 0.001,
                                               weight_decay = 0,
                                               device = device,
                                               save_name = 'trained_model_fold'+str(fold)+
                                               '_'+str(num_layers)+'layers.pt')
                                               # save_name = 'test_model'+str(num_splits)+'splits.pt')
            # trained_models.append(trained_model)
            break

    run_KFolds()


    # mean_norm_test, std_norm_test = calc_means_stds(original_images)
    #
    # final_model = cnn
    # final_model.load_state_dict(torch.load('models/trained_model.pt')['state_dict'])
