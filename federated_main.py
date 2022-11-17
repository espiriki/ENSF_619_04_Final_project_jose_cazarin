#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from cmath import sqrt
from torchvision import transforms
import torchvision
from models import *
from update import calculate_acc_global_dataset
from options import args_parser
import torch
import matplotlib.pyplot as plt
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import csv
import copy
import base_bin
from utils import average_weights
import keep_aspect_ratio

TRAIN_DATA_PATH = "./original_dataset_rgba"

eff_net_sizes = {
    'b0': (256, 224),
    'b4': (384, 380),
    'b7': (633, 600),
}


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
        args.lr = 0.01
    elif args.model == "res18":
        global_model = ResNet18()
        input_size = (300, 300)
        batch_size = 32
        args.lr = 0.01
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
        args.lr = 0.01
    elif args.model == "vision":
        global_model = VisionLarge32()
        input_size = (224, 224)
        batch_size = 24
        args.lr = 0.008

    print("Batch Size: {}".format(batch_size))
    print("Training for {} Global Epochs".format(args.epochs))
    print("And {} Local Epochs".format(args.local_ep))

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

    global_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)

    # If the batch size is too small, this means the model is too big, which limits
    # the batch size that can be used. To mitigate this problem, the number of workers
    # in the dataloader is limited, since they increase GPU mem usage
    _num_workers = 8
    if batch_size < 16:
        _num_workers = 4

    black_bin = base_bin.BaseBin(args, "./non_iid_dataset_rgba/black_bin",
                                 WIDTH, HEIGHT, batch_size, _num_workers, "Black bin", device)

    green_bin = base_bin.BaseBin(args, "./non_iid_dataset_rgba/green_bin",
                                 WIDTH, HEIGHT, batch_size, _num_workers, "Green bin", device)

    blue_bin = base_bin.BaseBin(args, "./non_iid_dataset_rgba/blue_bin",
                                WIDTH, HEIGHT, batch_size, _num_workers, "Blue bin", device)

    print("Starting training...")

    bins = {"black": black_bin, "blue": blue_bin, "green": green_bin}

    train_loss = []
    epoch_loss = []
    epoch_train_accuracy = []
    for epoch in range(args.epochs):
        batch_loss = []

        local_weights, local_losses = [], []
        print("Global Training Epoch : {}".format(epoch))

        for bin in bins:
            local_model = bins[bin]
            w_local_update, loss = local_model.local_update_weights(
                model=copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w_local_update))
            local_losses.append(copy.deepcopy(loss))

        print("Local training finished!\n")
        # update global weights

        print("Averaging weights of {} different models".format(len(local_weights)))
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # calculate global loss
        loss_avg_global = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg_global)

        global_weights_path = "./fed_global_model_weights/model_{}_epoch_{}.model".format(
            args.model, epoch+1)

        print("Saving global weights to {}".format(global_weights_path))

        torch.save(global_model.state_dict(), global_weights_path)

        print("Calculating Global accuracy...")
        train_accuracy = calculate_acc_global_dataset(
            global_model, global_dataset, 4, device)
        print("Global acc on global epoch {}: {:.3f}".format(
            epoch, train_accuracy))
        epoch_train_accuracy.append(train_accuracy)

        epoch_loss.append(loss_avg_global)
        print("Global loss on global epoch {}: {:.3f}".format(
            epoch, loss_avg_global))

    with open('fed_save/train_loss_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: x, epoch_loss))

    with open('fed_save/train_acc_{}.csv'.format(args.model), 'w') as f:

        write = csv.writer(f)
        write.writerow(map(lambda x: x, epoch_train_accuracy))

    # Plot loss
    epoch_loss = torch.FloatTensor(epoch_loss).cpu()
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.savefig(
        'fed_save/baseline_[M]_{}_[Global_E]_{}_[Local_Epochs]_[LR]_{}_loss.png'.format(args.model, args.epochs, args.local_ep, args.lr))

    plt.figure()
    plt.plot(range(len(epoch_train_accuracy)), epoch_train_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy per Epoch')
    plt.savefig(
        'fed_save/baseline_[M]_{}_[Global_E]_{}_[Local_Epochs]_{}_[LR]_{}_accuracy_per_epoch.png'.format(args.model, args.epochs, args.local_ep, args.lr))
