#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torchvision.models import *
import torch.nn as nn

def EffNetB4(num_classes = 4):

    model = efficientnet_b4(weights=None)
    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)   
    return model


def EffNetB0(num_classes = 4):

    model = efficientnet_b0(weights=None)
    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)   
    return model

def EffNetB7(num_classes = 4):

    model = efficientnet_b7(weights=None)
    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)   
    return model


def ResNet18(num_classes = 4):
    model = resnet18(weights=None)
    _in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model

def ResNet50(num_classes = 4):
    model = resnet50(weights=None)
    _in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model


def ResNet152(num_classes = 4):
    model = resnet152(weights=None)
    _in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model


def ConvNextTiny(num_classes = 4):
    model = convnext_tiny(weights=None)
    _in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model


def MBNetLarge(num_classes = 4):
    model = mobilenet_v3_large(weights=None)
    _in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model


def VisionLarge32(num_classes = 4):
    model = vit_l_32(weights=None)
    _in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model
