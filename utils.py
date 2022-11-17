#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weight_average_weights(w, weights_avg):
    """
    Returns the weighted average of the weights.
    """

    total_weight = np.sum(weights_avg)

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * weights_avg[0]

        for i in range(1, len(w)):
            w_avg[key] += (w[i][key] * weights_avg[i])

        w_avg[key] = torch.div(w_avg[key], total_weight)

    return w_avg
