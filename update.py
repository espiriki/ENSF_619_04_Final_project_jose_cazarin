#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math


def calculate_acc_global_dataset(model, dataset, _batch_size):

    device = "cuda:0"

    model.eval()
    total, correct = 0.0, 0.0
    # model.to(device)

    testloader = DataLoader(dataset, batch_size=_batch_size,
                            shuffle=True, num_workers=8,pin_memory=True)

    num_batches = math.ceil(
        (len(testloader.dataset)/_batch_size))

    for batch_idx, (images, labels) in enumerate(testloader):

        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        print("Batches {}/{} ".format(batch_idx,
                                      num_batches), end='\r')

    # model.to("cpu")
    accuracy = 100 * correct/total
    return accuracy
