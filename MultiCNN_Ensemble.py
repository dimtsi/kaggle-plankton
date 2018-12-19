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
from NNs import ResNetDynamic, ResNetMine, CNN, SuperNet, EnsembleClassifier, densenet201
from train_single import *
from train_single import ListsTrainDataset, ListsTestDataset

from torchsummary import summary


def predict_on_my_test_set(model, mean_norm_test, std_norm_test, multiGPU=False):

    test_transforms = transforms. Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_norm_test],
                    std =[std_norm_test])
    ])

    test_mine_dataset = ListsTrainDataset(test_mine_images,
                                          test_mine_labels,
                                          transform = test_transforms)
    test_mine_loader = torch.utils.data.DataLoader(test_mine_dataset,
                                                   batch_size = 32,
                                                   shuffle = False)

    best_accuracy = 0
    model.eval().to(device)
    if isinstance(model, EnsembleClassifier):
        if multiGPU == True:
            print("multiGPU")
            model.set_devices_multiGPU()

    correct = 0
    total = 0
    for images, labels in test_mine_loader:
        images = Variable(images)
        labels= labels.squeeze(1)
        outputs = model(images).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu().long() == labels).sum()
        test_accuracy = 100*correct.item() / total
    print('TEST SET ACCURACY: %.4f %%' % test_accuracy)
        # save_model(epoch, model, optimizer, scheduler)




if __name__ == "__main__":
    # print("weighted classes")
    classified = "extra"
    if classified == "full":
        original_images = pickle.load(open("pkl/train_padded64.pkl", "rb"))
        original_labels = pickle.load(open("pkl/train_labels.pkl", "rb"))
        test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes.pkl", "rb"))

    elif classified == "unknown_only":
        original_images = pickle.load(open("pkl/classified_padded64.pkl", "rb"))
        original_labels = pickle.load(open("pkl/classified_train_labels.pkl", "rb"))
        test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes_classified.pkl", "rb"))

    elif classified == "extra":
        original_images = pickle.load(open("pkl/extraclassified_padded64.pkl", "rb"))
        original_labels = pickle.load(open("pkl/extraclassified_train_labels.pkl", "rb"))
        test_set_mine_indexes = pickle.load(open("pkl/test_set_mine_indexes_extraclassified.pkl", "rb"))

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import timeit

    ##Class weights for imbalance
    # from sklearn.utils.class_weight import compute_class_weight
    # labels_df = pd.read_csv('train_onelabel.csv')
    # class_weights = compute_class_weight('balanced', np.arange(121), labels_df['class'])
    # class_weights = np.interp(class_weights, (class_weights.min(), class_weights.max()), (0, +1))
    # class_weights = torch.from_numpy(class_weights).float().to(device)

    from sklearn.model_selection import StratifiedKFold
    num_layers = 2
    models = []
    pretrained = resnet50(pretrained = True)
    # cnn1 = ResNetDynamic(pretrained.block, pretrained.layers,
    #             num_layers = 2, pretrained_nn = None)
    # cnn1_dict = torch.load('models/trained_model_fold0_'+str(num_layers)+'layers.pt')['state_dict']
    # cnn1.load_state_dict(cnn1_dict)
    # models.append(cnn1)
    #
    # cnn2 = ResNetDynamic(pretrained.block, pretrained.layers,
    #             num_layers = 2, pretrained_nn = None)
    # cnn2_dict = torch.load('models/trained_model_fold1_'+str(num_layers)+'layers.pt')['state_dict']
    # cnn2.load_state_dict(cnn2_dict)
    # models.append(cnn2)
    #
    # cnn3 = ResNetDynamic(pretrained.block, pretrained.layers,
    #             num_layers = 2, pretrained_nn = None)
    # cnn3_dict = torch.load('models/trained_model_fold2_'+str(num_layers)+'layers.pt')['state_dict']
    # cnn3.load_state_dict(cnn3_dict)
    # models.append(cnn3)
    #
    # cnn4 = ResNetDynamic(pretrained.block, pretrained.layers,
    #             num_layers = 2, pretrained_nn = None)
    # cnn4_dict = torch.load('models/trained_model_fold3_'+str(num_layers)+'layers.pt')['state_dict']
    # cnn4.load_state_dict(cnn4_dict)
    # models.append(cnn4)


    # cnn1 = ResNetDynamic(pretrained.block, pretrained.layers,
    #             num_layers = 2, pretrained_nn = None)
    # cnn1_dict = torch.load('models/trained_model_fold0_'+'stratified.pt')['state_dict']
    # cnn1.load_state_dict(cnn1_dict)
    # models.append(cnn1)
    #
    # cnn2 = ResNetDynamic(pretrained.block, pretrained.layers,
    #             num_layers = 2, pretrained_nn = None)
    # cnn2_dict = torch.load('models/trained_model_fold1_'+'stratified.pt')['state_dict']
    # cnn2.load_state_dict(cnn2_dict)
    # models.append(cnn2)
    #
    # cnn3 = ResNetDynamic(pretrained.block, pretrained.layers,
    #             num_layers = 2, pretrained_nn = None)
    # cnn3_dict = torch.load('models/trained_model_fold2_'+'stratified.pt')['state_dict']
    # cnn3.load_state_dict(cnn3_dict)
    # models.append(cnn3)
    #
    # cnn4 = ResNetDynamic(pretrained.block, pretrained.layers,
    #             num_layers = 2, pretrained_nn = None)
    # cnn4_dict = torch.load('models/trained_model_fold3_'+'stratified.pt')['state_dict']
    # cnn4.load_state_dict(cnn4_dict)
    # models.append(cnn4)

    cnn1 = ResNetDynamic(pretrained.block, pretrained.layers,
                num_layers = 2, pretrained_nn = None)
    cnn1_dict = torch.load('models/extraclassified/trained_model_3_new.pt')['state_dict']
    cnn1.load_state_dict(cnn1_dict)
    models.append(cnn1)

    cnn2 = ResNetDynamic(pretrained.block, pretrained.layers,
                num_layers = 2, pretrained_nn = None)
    cnn2_dict = torch.load('models/extraclassified/trained_model_15_new.pt')['state_dict']
    cnn2.load_state_dict(cnn2_dict)
    models.append(cnn2)

    cnn3 = ResNetDynamic(pretrained.block, pretrained.layers,
                num_layers = 2, pretrained_nn = None)
    cnn3_dict = torch.load('models/extraclassified/trained_model_90_new.pt')['state_dict']
    cnn3.load_state_dict(cnn3_dict)
    models.append(cnn3)


    # cnn4 = densenet201(pretrained=False)
    # cnn4_dict = torch.load('models/extraclassified/dense201.pt')['state_dict']
    # cnn4.load_state_dict(cnn4_dict)
    # models.append(cnn4)

    # cnn2 = ResNetDynamic(Bottleneck, [2, 2, 2, 3],num_layers = 4)
    cnn = EnsembleClassifier(models)

    def train_ensemble_on_whole_test_mine():
        norm = {}
        norm['train_norm_mean'], norm['train_norm_std'] = calc_means_stds(train_images)

        kf = StratifiedKFold(n_splits=4, random_state=None, shuffle=True)
        for additional_train_indexes_no_test, additional_validation_indexes_no_test in kf.split(X = test_mine_images,
                                                          y = test_mine_labels):
            additional_images = []
            additional_labels = []
            new_test_mine_images = []
            new_test_mine_labels = []

            for i in additional_train_indexes_no_test:
                additional_images.append(test_mine_images[i])
                additional_labels.append(test_mine_labels[i])
            for j in additional_validation_indexes_no_test:
                new_test_mine_images.append(test_mine_images[j])
                new_test_mine_labels.append(test_mine_labels[j])


        extended_train_images = (train_images_no_test+additional_images).copy()
        extended_train_labels = (train_labels_no_test+additional_labels).copy()

        train_dataset, val_dataset = create_train_val_datasets(extended_train_images,
                                                               extended_train_labels,
                                                               new_test_mine_images,
                                                               new_test_mine_labels,
                                                               norm_params = norm)

        print("train: "+ str(len(extended_train_images))
              + " val: " +str(len(new_test_mine_images)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,
            shuffle = True, num_workers=4)

        test_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size = 32, shuffle = False)

        # cnn.to(device)
        trained_model = train_and_validate(cnn, train_loader, test_loader,
                                           num_epochs=200, device = device,
                                           multiGPU = True,
                                           save_name = 'extraclassified/ensemble_1fc.pt')

    # train_ensemble_on_whole_test_mine()


    def train_ensemble_on_test():
        norm = {}
        norm['train_norm_mean'], norm['train_norm_std'] = calc_means_stds(original_images)
        train_dataset, val_dataset = create_train_val_datasets(train_images_no_test,
                                                               train_labels_no_test,
                                                               test_mine_images,
                                                               test_mine_labels,
                                                               norm_params = norm)

        # train_sampler = ImbalancedDatasetSampler(train_dataset)
        print("train: "+ str(len(train_images))
              + " val: " +str(len(test_mine_images)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,
            shuffle = True, num_workers=4)

        test_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size = 32, shuffle = False)

        # cnn.to(device)
        trained_model = train_and_validate(cnn, train_loader, test_loader,
                                           num_epochs=200, device = device,
                                           multiGPU = True,
                                           save_name = 'extraclassified/ensemble_1fc.pt')
    print(cnn)
    train_ensemble_on_test()
    #
    mean_norm_test, std_norm_test = calc_means_stds(train_images)

    final_model = cnn.to(device)
    final_model.load_state_dict(torch.load('models/extraclassified/ensemble_2fc.pt')['state_dict'])

    # predict_on_my_test_set(final_model, mean_norm_test, std_norm_test, multiGPU=False)
    predict_test_set_kaggle(final_model, test_filenames,
                            mean_norm_test, std_norm_test,
                            multiGPU=True)
