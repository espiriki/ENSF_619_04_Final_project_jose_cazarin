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
from torchsummary import summary
import albumentations as A
import cv2
import albumentations.pytorch as a_pytorch
import numpy as np
import wandb
import torch.nn as nn

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

def run_one_epoch(epoch, global_model, data_loader_train, len_train_data, device, batch_size):

    batch_loss = []
    num_batches = math.ceil((len_train_data/batch_size))

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

def calculate_train_accuracy(global_model, data_loader_train, len_train_data, device, batch_size):

    correct = 0.0
    num_batches = math.ceil((len_train_data/batch_size))

    with torch.no_grad():
        
        for batch_idx, (images, labels) in enumerate(data_loader_train):

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
    train_accuracy = 100 * (correct/len(train_data))

    return train_accuracy

def calculate_val_accuracy(global_model, data_loader_val, len_val_data, device, batch_size):

    correct = 0.0
    num_batches = math.ceil((len_val_data/batch_size))
    with torch.no_grad():

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
    val_accuracy = 100 * (correct/len(train_data))
    return val_accuracy

def save_model_weights(global_model,args, epoch):

    weights_path = BASE_PATH + "model_weights/model_{}_epoch_{}.model".format(
    args.model, epoch+1)

    print("Saving weights to {}".format(weights_path))

    torch.save(global_model.state_dict(), weights_path)

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
    _batch_size = 32
    if args.model == "b4":
        global_model = EffNetB4()
        input_size = eff_net_sizes[args.model]
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
    else:
        print("Invalid Model: {}".format(args.model))
        sys.exit(1)

    print("Batch Size: {}".format(_batch_size))
    print("Learning Rate: {}".format(args.lr))
    print("Training for {} epochs".format(args.epochs))

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        global_model = nn.DataParallel(global_model)

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

    train_data, val_data = random_split(train_data, [int(
        math.ceil(len(train_data)*0.8)), int(math.floor(len(train_data)*0.2))])

    print("Num of training images: {}".format(len(train_data)))
    print("Num of validaton images: {}".format(len(val_data)))

    optimizer = torch.optim.AdamW(global_model.parameters(), lr=args.lr)

    # cluster says the recommended ammount is 8
    _num_workers = 8

    data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=_batch_size,
                                                    shuffle=True, num_workers=_num_workers, pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset=val_data,
                                                  batch_size=_batch_size,
                                                  shuffle=True, num_workers=_num_workers, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    print("Starting training...")
    global_model.to(device)
    for epoch in range(args.epochs):
        
        global_model.train()
        num_batches, train_loss_per_batch = run_one_epoch(epoch,
                                                          global_model,
                                                          data_loader_train,
                                                          len(train_data),
                                                          device,
                                                          _batch_size)

        train_loss_avg = sum(train_loss_per_batch)/len(train_loss_per_batch)
        train_loss_history.append(train_loss_avg)
        print("Avg train loss on epoch {}: {:.3f}".format(epoch, train_loss_avg))

        global_model.eval()

        print("Starting train accuracy calculation for epoch {}".format(epoch))
        train_accuracy = calculate_train_accuracy(global_model,
                                                  data_loader_train,
                                                  len(train_data), 
                                                  device,
                                                  _batch_size)

        print("Train set accuracy on epoch {}: {:.3f} ".format(epoch, train_accuracy))
        train_accuracy_history.append(train_accuracy)

        print("Starting validation accuracy calculation for epoch {}".format(epoch))
        val_accuracy = calculate_val_accuracy(global_model,
                                              data_loader_val,
                                              len(val_data),
                                              device,
                                              _batch_size)

        print("Val set accuracy on epoch {}: {:.3f}".format(epoch, val_accuracy))
        val_accuracy_history.append(val_accuracy)

        save_model_weights(global_model, args, epoch)


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
    plt.ylabel('Val accuracy')
    plt.savefig(
        BASE_PATH + 'save/[M]_{}_[E]_{}_[LR]_{}_val_accuracy.png'.format(args.model, args.epochs, args.lr))
