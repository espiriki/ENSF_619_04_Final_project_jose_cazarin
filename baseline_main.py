#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from cmath import sqrt
from torchvision import transforms
import torchvision
from models import *
from update import test_inference
from options import args_parser
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import os
import math
import csv
import keep_aspect_ratio

eff_net_sizes = {
    'b0': (256, 224),
    'b4': (384, 380),
    'b7': (633, 600),
}

TRAIN_DATA_PATH = "./original_dataset_rgba"


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


if __name__ == '__main__':
    args = args_parser()

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        torch.cuda.empty_cache()
        free_gpu_cache()
        print("GPU OK!!!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Model: {}".format(args.model))

    # Those max batch sizes were based on 8GB of GPU memory
    # which is what I have in my local PC
    global_model = EffNetB4()
    input_size = eff_net_sizes["b4"]
    batch_size = 10
    if args.model == "b4":
        global_model = EffNetB4()
        input_size = eff_net_sizes[args.model]
        batch_size = 8
    elif args.model == "b0":
        global_model = EffNetB0()
        input_size = eff_net_sizes[args.model]
        batch_size = 32
    elif args.model == "res18":
        global_model = ResNet18()
        input_size = (300, 300)
        batch_size = 32
    elif args.model == "res50":
        global_model = ResNet50()
        input_size = (400, 400)
        batch_size = 16
    elif args.model == "res152":
        global_model = ResNet152()
        input_size = (500, 500)
        batch_size = 6
    elif args.model == "next_tiny":
        global_model = ConvNextTiny()
        input_size = (224, 224)
        batch_size = 32
    elif args.model == "mb":
        global_model = MBNetLarge()
        input_size = (320, 320)
        batch_size = 32
    elif args.model == "vision":
        global_model = VisionLarge32()
        input_size = (224, 224)
        batch_size = 24

    print("Batch Size: {}".format(batch_size))
    print("Training for {} epochs".format(args.epochs))

    WIDTH = input_size[0]
    HEIGHT = input_size[1]
    AR_INPUT = WIDTH / HEIGHT

    TRANSFORM_IMG = transforms.Compose([
        transforms.RandomRotation(degrees=(-90, 90), expand=True),
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        transforms.Resize(
            (WIDTH, HEIGHT), transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAutocontrast(),
        transforms.RandomPerspective(),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ToTensor(),
        # Those per-channel mean and std values were obtained using the
        # calculate_mean_std_dataset.py script
        transforms.Normalize([0.5599, 0.5358, 0.5033],
                             [0.3814, 0.3761, 0.3833]),
    ])

    train_data = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)

    # Set the model to train and send it to device.
    global_model.to(device)

    print("Learning Rate: {}".format(args.lr))
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)

    # If the batch size is too small, this means we are running out of GPU mem
    # so limit the number of workers as well, which increases GPU mem usage
    _num_workers = 8
    if batch_size < 16:
        _num_workers = 4

    data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True, num_workers=_num_workers)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []
    epoch_train_accuracy = []

    print("Starting training...")
    for epoch in range(args.epochs):
        batch_loss = []

        correct = 0
        num_batches = math.ceil((len(train_data)/batch_size))

        global_model.train()

        for batch_idx, (images, labels) in enumerate(data_loader_train):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = global_model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} on Epoch {}".format(batch_idx,
                  num_batches, epoch), end='\r')

            batch_loss.append(loss.detach())

        weights_path = "./model_weights/model_{}_epoch_{}.model".format(
            args.model, epoch+1)

        print("Saving weights to {}".format(weights_path))

        torch.save(global_model.state_dict(), weights_path)

        train_accuracy = 100 * correct / len(train_data)
        print("Train set acc on epoch {}: {:.3f}".format(epoch, train_accuracy))
        epoch_train_accuracy.append(train_accuracy)

        loss_avg = sum(batch_loss)/len(batch_loss)
        epoch_loss.append(loss_avg)
        print("Loss on epoch {}: {:.3f}".format(epoch, loss_avg))

    with open('save/train_loss_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], epoch_loss))

    with open('save/train_acc_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], epoch_train_accuracy))

    # Plot loss
    epoch_loss = torch.FloatTensor(epoch_loss).cpu()
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.savefig(
        'save/baseline_[M]_{}_[E]_{}_[LR]_{}_loss.png'.format(args.model, args.epochs, args.lr))

    plt.figure()
    plt.plot(range(len(epoch_train_accuracy)), epoch_train_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy per Epoch')
    plt.savefig(
        'save/baseline_[M]_{}_[E]_{}_[LR]_{}_accuracy_per_epoch.png'.format(args.model, args.epochs, args.lr))
