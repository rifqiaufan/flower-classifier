import torch
from torch import nn, optim
from collections import OrderedDict

def update_classifier(model,hidden_units=512):
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1',nn.Dropout(0.3)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    return model