#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torchvision.models import *


def EffNetB7():

    return efficientnet_b7(weights=None)


def EffNetB4():

    return efficientnet_b4(weights=None)


def EffNetB0():

    return efficientnet_b0(weights=None)
