#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from cmath import sqrt
from torchvision import transforms
import torchvision
import sys
# from torchsummary import summary
from models import *
from update import test_inference
from options import args_parser
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import gc
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import os
import math
import csv
import keep_aspect_ratio

TRAIN_DATA_PATH = "./original_dataset_rgba"

WIDTH = 384
HEIGHT = 380
AR_INPUT = WIDTH / HEIGHT

TRANSFORM_IMG = transforms.Compose([
    transforms.RandomRotation(degrees=(-90, 90), expand=True),
    keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
    transforms.Resize((WIDTH, HEIGHT), transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAutocontrast(),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize([0.6975, 0.6666, 0.6239], [0.2863, 0.2918, 0.3213]),
])


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def calculate_val_acc(global_model, data_loader_val, len_val_set, device):

    global_model.eval()
    correct = 0

    with torch.no_grad():

        for _, (images, labels) in enumerate(data_loader_val):

            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

    return 100 * (correct/len_val_set)


if __name__ == '__main__':
    args = args_parser()

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        torch.cuda.empty_cache()
        free_gpu_cache()
        print("GPU OK!!!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    global_model = EffNetB4()
    model_weights = "model_save/epoch_25_batch_800.model"

    if os.path.exists(model_weights):
        print("Loading weights {}".format(model_weights))
        global_model.load_state_dict(torch.load(model_weights))
    else:
        print("Starting from scratch")

    train_data = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)

    n = len(train_data)  # total number of examples
    n_train = int(0.7 * n)  # take ~70% for test
    n_val = int(0.2 * n)  # take ~20% for test
    n_test = int(0.10 * n)  # take ~10% for test
    train_set = torch.utils.data.Subset(
        train_data, range(n_train))  # take first 70%
    val_set = torch.utils.data.Subset(
        train_data, range(n_train, n_train + n_val))  # take the rest
    test_set = torch.utils.data.Subset(
        train_data, range(n_train + n_val, n))  # take the rest

    print(train_data.classes)
    print(train_data.class_to_idx)

    print(len(train_set))
    print(len(val_set))
    print(len(test_set))

    # Set the model to train and send it to device.
    global_model.to(device)
    # print(global_model)

    print("Learning Rate: {}".format(args.lr))
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)

    _batch_size = 10

    data_loader_train = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=_batch_size,
                                                    shuffle=True, num_workers=int(_batch_size/2))

    data_loader_val = torch.utils.data.DataLoader(dataset=val_set,
                                                  batch_size=int(
                                                      _batch_size),
                                                  shuffle=True, num_workers=int(_batch_size/2))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []
    train_accuracy_history = []
    val_accuracy_history = []

    print("Starting training...")
    for epoch in range(args.epochs):
        batch_loss = []

        correct = 0
        num_batches = math.ceil((len(train_set)/_batch_size))

        global_model.train()

        for batch_idx, (images, labels) in enumerate(data_loader_train):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # print("Calculating inferences...")
            outputs = global_model(images)

            # print("Calculating loss...")
            loss = criterion(outputs, labels)

            # print("Backpropagation...")
            loss.backward()

            # print("Updating optimizer")
            optimizer.step()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} on Epoch {}".format(batch_idx,
                  num_batches, epoch), end='\r')

            batch_loss.append(loss.detach())

            if batch_idx % 50 == 0 and batch_idx != 0:
                print("Loss: {}".format(loss.item()))
                weights_path = "./model_save/epoch_" + \
                    str(epoch+1)+"_batch_"+str(batch_idx)+".model"

                print("saving weights to {}".format(weights_path))

                torch.save(global_model.state_dict(), weights_path)

        train_accuracy = 100 * correct / len(train_set)
        print("Train set acc on epoch {}: {:.3f}".format(epoch, train_accuracy))
        train_accuracy_history.append(train_accuracy)

        val_accuracy = calculate_val_acc(
            global_model, data_loader_val, len(val_set), device)

        print("Val set acc on epoch {}: {:.3f}".format(epoch, val_accuracy))
        val_accuracy_history.append(val_accuracy)

        loss_avg = sum(batch_loss)/len(batch_loss)
        epoch_loss.append(loss_avg)

    with open('save/val_acc.csv', 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], val_accuracy_history))

    with open('save/train_loss.csv', 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], epoch_loss))

    with open('save/train_acc.csv', 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], train_accuracy_history))

    # Plot loss

    epoch_loss = torch.FloatTensor(epoch_loss).cpu()
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.savefig('save/baseline_{}_{}_{}_loss.png'.format(args.dataset, args.model,
                                                         args.epochs))

    plt.figure()
    plt.plot(range(len(train_accuracy_history)), train_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy per Epoch')
    plt.savefig('save/train_baseline_accuracy_per_epoch.png'.format())

    plt.figure()
    plt.plot(range(len(val_accuracy_history)), val_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Val accuracy per Epoch')
    plt.savefig('save/val_baseline_accuracy_per_epoch.png'.format())
