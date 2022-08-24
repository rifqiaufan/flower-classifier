import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def preprocess(path):

#     data_dir = 'flowers'
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    image_train = datasets.ImageFolder(train_dir, transform = train_transforms)
    image_test = datasets.ImageFolder(test_dir, transform = test_transforms)
    image_validation = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    dataloaders_train = torch.utils.data.DataLoader(image_train, batch_size = 64, shuffle = True)
    dataloaders_test = torch.utils.data.DataLoader(image_test, batch_size = 64, shuffle = True)
    dataloaders_validation = torch.utils.data.DataLoader(image_validation, batch_size = 64, shuffle = True)
    
    return dataloaders_train, dataloaders_test, dataloaders_validation, image_train