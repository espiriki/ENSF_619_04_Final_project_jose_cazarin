#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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
from torchsummary import summary
import albumentations as A
import cv2
import albumentations.pytorch as a_pytorch
import numpy as np


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
    'b7': (633, 600),
}

BASE_PATH = "/project/def-rmsouza/jocazar/"

TRAIN_DATA_PATH = BASE_PATH + "original_dataset_rgba"


def print_summary(model, train_data):

    trainloader = DataLoader(train_data, batch_size=1, shuffle=True)

    for _, (image, _) in enumerate(trainloader):
        summary(model, image)
        break


def free_gpu_cache():
    print("Initial GPU Usage")

    print("GPU Usage after emptying the cache")


def run_one_epoch(epoch, global_model, data_loader_train, len_train_data, device):

    batch_loss = []
    num_batches = math.ceil((len_train_data/batch_size))

    global_model.to(device)

    print("Using device: {}".format(device))
    for batch_idx, (images, labels) in enumerate(data_loader_train):

        images, labels = images.to(device), labels.to(device)

        model_outputs = global_model(images)
        loss = criterion(model_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Batches {}/{} on epoch {}".format(batch_idx,
                                                 num_batches, epoch), end='\r')

        batch_loss.append(loss.detach())

    print("\n")

    return num_batches, batch_loss


if __name__ == '__main__':
    args = args_parser()

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Model: {}".format(args.model))

    # Those max batch sizes were based on 8GB of GPU memory
    # which is what I have in my local PC
    global_model = EffNetB4()
    input_size = eff_net_sizes["b4"]
    batch_size = 32
    if args.model == "b4":
        global_model = EffNetB4()
        input_size = eff_net_sizes[args.model]
    elif args.model == "b7":
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
    else:
        print("Invalid Model: {}".format(args.model))
        sys.exit(1)

    print("Batch Size: {}".format(batch_size))
    print("Training for {} epochs".format(args.epochs))

    WIDTH = input_size[0]
    HEIGHT = input_size[1]
    AR_INPUT = WIDTH / HEIGHT

    TRANSFORM_IMG = A.Compose([
        A.SafeRotate(p=1.0, interpolation=cv2.INTER_CUBIC,
                     border_mode=cv2.BORDER_CONSTANT,
                     value=0),
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_CUBIC),
        A.Flip(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.Sharpen(p=1.0),
        A.Perspective(p=1.0, fit_output=True,
                      keep_size=True,
                      pad_mode=cv2.BORDER_CONSTANT,
                      pad_val=0),
        A.Normalize([0.5599, 0.5358, 0.5033],
                    [0.3814, 0.3761, 0.3833]),
        a_pytorch.transforms.ToTensorV2()
    ])

    train_data = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH, transform=Transforms(img_transf=TRANSFORM_IMG))

    # train_data = Subset(train_data, range(1024))

    train_data, validation_data = random_split(train_data, [int(
        math.ceil(len(train_data)*0.8)), int(math.floor(len(train_data)*0.2))])

    print("Num of training images: {}".format(len(train_data)))
    print("Learning Rate: {}".format(args.lr))

    optimizer = torch.optim.AdamW(global_model.parameters(), lr=args.lr)

    # If the batch size is too small, this means we are running out of GPU mem
    # so limit the number of workers as well, which increases GPU mem usage
    _num_workers = 8

    data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True, num_workers=_num_workers, pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset=validation_data,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=_num_workers, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []
    epoch_val_accuracy = []

    print("Starting training...")

    batch_idx = 0
    for epoch in range(args.epochs):
        # Set the model to train
        global_model.train()
        num_batches, batch_loss = run_one_epoch(epoch,
                                                global_model,
                                                data_loader_train,
                                                len(train_data),
                                                device)

        global_model.eval()
        correct = 0.0
        print("Starting validation for epoch {}".format(epoch))
        num_batches = math.ceil((len(validation_data)/batch_size))
        for batch_idx, (images, labels) in enumerate(data_loader_val):

            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = global_model(images)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                          num_batches), end='\r')

        print("\n")
        val_accuracy = 100 * (correct/len(validation_data))

        print("Val set acc on epoch {}: {:.3f}".format(epoch, val_accuracy))
        epoch_val_accuracy.append(val_accuracy)

        loss_avg = sum(batch_loss)/len(batch_loss)
        epoch_loss.append(loss_avg)
        print("Loss on epoch {}: {:.3f}".format(epoch, loss_avg))

        weights_path = BASE_PATH + "model_weights/model_{}_epoch_{}.model".format(
            args.model, epoch+1)

        print("Saving weights to {}".format(weights_path))

        torch.save(global_model.state_dict(), weights_path)

        batch_idx = batch_idx + 1

    with open(BASE_PATH + 'save/train_loss_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], epoch_loss))

    with open(BASE_PATH + 'save/train_acc_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: [x], epoch_val_accuracy))

    # Plot loss
    epoch_loss = torch.FloatTensor(epoch_loss).cpu()
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.savefig(
        BASE_PATH + 'save/baseline_[M]_{}_[E]_{}_[LR]_{}_loss.png'.format(args.model, args.epochs, args.lr))

    plt.figure()
    plt.plot(range(len(epoch_val_accuracy)), epoch_val_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy per Epoch')
    plt.savefig(
        BASE_PATH + 'save/baseline_[M]_{}_[E]_{}_[LR]_{}_accuracy_per_epoch.png'.format(args.model, args.epochs, args.lr))
