#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torchvision.models import *


def EffNetB4():

    return efficientnet_b4(weights=None)


def EffNetB0():

    return efficientnet_b0(weights=None)


def ResNet18():
    return resnet18(weights=None)


def ResNet50():
    return resnet50(weights=None)


def ResNet152():
    return resnet152(weights=None)


def ConvNextTiny():
    return convnext_tiny(weights=None)


def MBNetLarge():
    return mobilenet_v3_large(weights=None)


def VisionLarge32():
    return vit_l_32(weights=None)
