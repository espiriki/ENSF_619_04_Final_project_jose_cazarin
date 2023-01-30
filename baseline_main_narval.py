#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from torchvision import transforms
import torchvision
from models import *
from options import args_parser
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
import torch
import matplotlib.pyplot as plt
import math
import csv
import keep_aspect_ratio
import albumentations as A
import cv2
import albumentations.pytorch as a_pytorch
import numpy as np
import wandb
import torch.nn as nn
import itertools
from itertools import groupby


class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image


eff_net_sizes = {
    'b0': (256, 224),
    'b4': (384, 380),
    'b5': (489, 456),
    'b6': (561, 528),
    'b7': (633, 600),
}

BASE_PATH = "/project/def-rmsouza/jocazar/"

TRAIN_DATA_PATH = BASE_PATH + "original_dataset_rgba"


def run_one_epoch(epoch_num, model, data_loader, len_train_data, hw_device, batch_size):

    batch_loss = []
    n_batches = math.ceil((len_train_data/batch_size))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.reg)
    criterion = torch.nn.CrossEntropyLoss().to(hw_device)

    print("Using device: {}".format(hw_device))
    for batch_idx, (images, labels) in enumerate(data_loader):

        images, labels = images.to(hw_device), labels.to(hw_device)

        model_outputs = model(images)
        loss = criterion(model_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Batches {}/{} on epoch {}".format(batch_idx,
                                                 n_batches, epoch_num), end='\r')

        cpu_loss = loss.cpu()
        cpu_loss = cpu_loss.detach()
        batch_loss.append(cpu_loss)

    print("\n")

    return n_batches, batch_loss


def calculate_train_accuracy(model, data_loader, len_train_data, hw_device, batch_size):

    correct = 0
    n_batches = math.ceil((len_train_data/batch_size))

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(data_loader):

            images, labels = images.to(hw_device), labels.to(hw_device)

            # Inference
            outputs = model(images)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                          n_batches), end='\r')

    print("\n")
    train_acc = 100 * (correct/len_train_data)
    return train_acc


def calculate_val_accuracy(model, data_loader, len_val_data, hw_device, batch_size):

    correct = 0
    n_batches = math.ceil((len_val_data/batch_size))
    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(data_loader):

            images, labels = images.to(hw_device), labels.to(hw_device)

            # Inference
            outputs = model(images)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                          n_batches), end='\r')

    print("\n")
    print("samples checked for val: {}".format(len_val_data))
    print("correct samples for val: {}".format(correct))
    val_acc = 100 * (correct/len_val_data)
    return val_acc


def save_model_weights(model, model_name, epoch_num, val_acc):

    weights_path = BASE_PATH + "model_weights/BEST_model_{}_epoch_{}_LR_{}_Reg_{}_VAL_ACC_{}_.model".format(
        model_name, epoch_num+1, args.lr, args.reg, val_acc)

    print("Saving weights to {}".format(weights_path))

    torch.save(model.state_dict(), weights_path)


def calculate_mean_std_train_dataset():

    ORIGINAL_IMAGE = A.Compose([
        a_pytorch.transforms.ToTensorV2()
    ])

    my_data = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH, transform=Transforms(img_transf=ORIGINAL_IMAGE))

    train, _ = random_split(my_data, [int(
        math.ceil(len(my_data)*0.8)), int(math.floor(len(my_data)*0.2))], generator=torch.Generator().manual_seed(42))

    channels_sum = 0
    channels_squared_sum = 0
    num_images = 0
    for element in train:
        image = element[0]
        channels_sum += torch.mean(image*1.0, dim=[1, 2])
        channels_squared_sum += torch.mean((image*1.0)**2, dim=[1, 2])
        num_images += 1

    mean = (channels_sum / num_images)/255

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (torch.sqrt((channels_squared_sum / num_images) - mean ** 2))/255

    print(mean)
    print(std)

    return 0, 0


if __name__ == '__main__':
    args = args_parser()

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Model: {}".format(args.model))

    global_model = EffNetB4()
    input_size = eff_net_sizes["b4"]
    _batch_size = 32
    if args.model == "b4":
        global_model = EffNetB4()
        input_size = eff_net_sizes[args.model]
    elif args.model == "b5":
        global_model = EffNetB5()
        input_size = eff_net_sizes[args.model]
        _batch_size = 16
    elif args.model == "b7":
        _batch_size = 8
        global_model = EffNetB7()
        input_size = eff_net_sizes[args.model]
    elif args.model == "b0":
        global_model = EffNetB0()
        input_size = eff_net_sizes[args.model]
    elif args.model == "res18":
        global_model = ResNet18()
        input_size = (300, 300)
    elif args.model == "res50":
        global_model = ResNet50()
        input_size = (400, 400)
    elif args.model == "res152":
        global_model = ResNet152()
        input_size = (500, 500)
    elif args.model == "next_tiny":
        global_model = ConvNextTiny()
        input_size = (224, 224)
    elif args.model == "mb":
        global_model = MBNetLarge()
        input_size = (320, 320)
    elif args.model == "vision":
        global_model = VisionLarge32()
        input_size = (224, 224)
    elif args.model == "visionb":
        global_model = VisionB32()
        input_size = (224, 224)
    else:
        print("Invalid Model: {}".format(args.model))
        sys.exit(1)

    print(global_model)

    print("Batch Size: {}".format(_batch_size))
    print("Learning Rate: {}".format(args.lr))
    print("Training for {} epochs".format(args.epochs))
    print("Regularization Rate: {}".format(args.reg))

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        global_model = nn.DataParallel(global_model)

    WIDTH = input_size[0]
    HEIGHT = input_size[1]
    AR_INPUT = WIDTH / HEIGHT

    prob_augmentations = 0.5

    # mean_train_dataset, std_train_dataset = calculate_mean_std_train_dataset()
    mean_train_dataset = [0.6958, 0.6680, 0.6307]
    std_train_dataset = [0.2790, 0.3010, 0.3273]

    normalize_transform = A.Normalize(mean_train_dataset,
                                      std_train_dataset)

    TRANSFORM_IMG = A.Compose([
        A.SafeRotate(p=prob_augmentations, interpolation=cv2.INTER_CUBIC,
                     border_mode=cv2.BORDER_CONSTANT,
                     value=0),
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_CUBIC),
        A.VerticalFlip(p=prob_augmentations),
        A.HorizontalFlip(p=prob_augmentations),
        A.RandomBrightnessContrast(p=prob_augmentations),
        # A.Sharpen(p=prob_augmentations),
        A.Perspective(p=prob_augmentations, fit_output=True,
                      keep_size=True,
                      pad_mode=cv2.BORDER_CONSTANT,
                      pad_val=0),
        # A.Blur(p=prob_augmentations),
        # A.Downscale(p=prob_augmentations,
        #             interpolation=dict(downscale=cv2.INTER_CUBIC,
        #                                upscale=cv2.INTER_CUBIC)),
        # A.GaussNoise(p=prob_augmentations, per_channel=True),
        # A.RandomFog(p=prob_augmentations),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    NO_AUG = A.Compose([
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_CUBIC),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    all_data_img_folder = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH, transform=Transforms(img_transf=TRANSFORM_IMG))

    val_data_img_folder = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH, transform=Transforms(img_transf=NO_AUG))

    # all_data_img_folder = Subset(all_data_img_folder, range(20))
    # val_data_img_folder = Subset(val_data_img_folder, range(20))

    print("Splitting the dataset per class...")

    per_class_splitted_all_data = [[], [], [], []]
    per_class_splitted_val_data = [[], [], [], []]

    for element_all, element_val in zip(all_data_img_folder, val_data_img_folder):
        per_class_splitted_all_data[element_all[1]].append(element_all)
        per_class_splitted_val_data[element_val[1]].append(element_val)

    VALID_SPLIT = 0.8

    train_data_per_class = []
    val_data_per_class = []
    for all_data, val_data in zip(per_class_splitted_all_data, per_class_splitted_val_data):

        all_data_class = all_data
        val_data_class = val_data

        class_dataset_size = len(all_data_class)
        valid_size = int(VALID_SPLIT*class_dataset_size)
        indices = torch.randperm(class_dataset_size).tolist()
        dataset_val = Subset(all_data_class, indices[:-valid_size])
        dataset_train = Subset(val_data_class, indices[-valid_size:])
        train_data_per_class.append(dataset_train)
        val_data_per_class.append(dataset_val)

    train_data = list(itertools.chain.from_iterable(train_data_per_class))

    val_data = list(itertools.chain.from_iterable(val_data_per_class))

    # per_class_splitted_all_data = [list(g)
    #                                for _, g in groupby(all_data_img_folder, lambda x: x[1])]

    # train_data_class_0, val_data_class_0 = random_split(all_data_class_0, [int(
    #     math.ceil(len(all_data_class_0)*0.8)), int(math.floor(len(all_data_class_0)*0.2))],
    #     generator=torch.Generator().manual_seed(42))

    # train_data_class_1, val_data_class_1 = random_split(all_data_class_1, [int(
    #     math.ceil(len(all_data_class_1)*0.8)), int(math.floor(len(all_data_class_1)*0.2))],
    #     generator=torch.Generator().manual_seed(42))

    # train_data_class_2, val_data_class_2 = random_split(all_data_class_2, [int(
    #     math.ceil(len(all_data_class_2)*0.8)), int(math.floor(len(all_data_class_2)*0.2))],
    #     generator=torch.Generator().manual_seed(42))

    # train_data_class_3, val_data_class_3 = random_split(all_data_class_3, [int(
    #     math.ceil(len(all_data_class_3)*0.8)), int(math.floor(len(all_data_class_3)*0.2))],
    #     generator=torch.Generator().manual_seed(42))

    # all_data = train_data_class_0 + train_data_class_1 + \
    #     train_data_class_2 + train_data_class_3
    # val_data = val_data_class_0 + val_data_class_1 + \
    #     val_data_class_2 + val_data_class_3

    print("Num of training images: {}".format(len(train_data)))
    print("Num of validaton images: {}".format(len(val_data)))

    # cluster says the recommended ammount is 8
    _num_workers = 8

    data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=_batch_size,
                                                    shuffle=True, num_workers=_num_workers, pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset=val_data,
                                                  batch_size=_batch_size,
                                                  shuffle=True, num_workers=_num_workers, pin_memory=True)

    train_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    print("Starting training...")
    global_model.to(device)
    max_val_accuracy = 0.0

    for epoch in range(args.epochs):

        global_model.train()
        num_batches, train_loss_per_batch = run_one_epoch(epoch,
                                                          global_model,
                                                          data_loader_train,
                                                          len(train_data),
                                                          device,
                                                          _batch_size)

        train_loss_avg = np.average(train_loss_per_batch)
        train_loss_history.append(train_loss_avg)

        print("Avg train loss on epoch {}: {:.3f}".format(epoch, train_loss_avg))
        print("Max train loss on epoch {}: {:.3f}".format(
            epoch, np.max(train_loss_per_batch)))
        print("Min train loss on epoch {}: {:.3f}".format(
            epoch, np.min(train_loss_per_batch)))

        global_model.eval()

        print("Starting train accuracy calculation for epoch {}".format(epoch))
        train_accuracy = calculate_train_accuracy(global_model,
                                                  data_loader_train,
                                                  len(train_data),
                                                  device,
                                                  _batch_size)

        print("Train set accuracy on epoch {}: {:.3f} ".format(
            epoch, train_accuracy))
        train_accuracy_history.append(train_accuracy)

        print("Starting validation accuracy calculation for epoch {}".format(epoch))
        val_accuracy = calculate_val_accuracy(global_model,
                                              data_loader_val,
                                              len(val_data),
                                              device,
                                              _batch_size)

        print("Val set accuracy on epoch {}: {:.3f}".format(epoch, val_accuracy))
        val_accuracy_history.append(val_accuracy)

        if val_accuracy > max_val_accuracy:
            print("Best model obtained based on Val Acc. Saving it!")
            save_model_weights(global_model, args.model, epoch, val_accuracy)
            max_val_accuracy = val_accuracy
        else:
            print("Not saving model, best Val Acc so far: {:.3f}".format(
                max_val_accuracy))

    # Finished training, save data
    with open(BASE_PATH + 'save/train_loss_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], train_loss_history))

    with open(BASE_PATH + 'save/train_acc_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], train_accuracy_history))

    with open(BASE_PATH + 'save/val_acc_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], val_accuracy_history))

    # Plot train loss
    train_loss_history = torch.FloatTensor(train_loss_history).cpu()
    plt.figure()
    plt.plot(range(len(train_loss_history)), train_loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_train_loss.png'.format(args.model, args.epochs, args.lr))

    # Plot train accuracy
    train_accuracy_history = torch.FloatTensor(train_accuracy_history).cpu()
    plt.figure()
    plt.plot(range(len(train_accuracy_history)), train_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy')
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_train_accuracy.png'.format(args.model, args.epochs, args.lr))

    # Plot val accuracy
    plt.figure()
    plt.plot(range(len(val_accuracy_history)), val_accuracy_history)
    plt.xlabel('Epochs')
    plt.ylabel('Val accuracy per Epoch')
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_val_accuracy.png'.format(args.model, args.epochs, args.lr))
