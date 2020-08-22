# import torchvision
import torch 
import torch.nn as nn
from torchvision import models


def build_model():
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False

    resnet.fc = nn.Sequential(nn.Linear(resnet.fc.in_features, 500),
                         nn.ReLU(),
                         nn.Dropout(), nn.Linear(500, 264))

    return resnet