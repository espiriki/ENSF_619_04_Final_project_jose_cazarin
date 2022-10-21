#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from cmath import sqrt
from torchvision import transforms
import torchvision
import sys
from torchsummary import summary
from models import EffNetB7
from update import test_inference
from options import args_parser
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
TRAIN_DATA_PATH = "original_dataset_rgba"


TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((633, 600), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])


def mean_std(loader):
    images, _ = next(iter(loader))
    print("batch size 1: {}".format(len(images)))
    mean = torch.mean(images, dim=[0, 2, 3])
    std = torch.std(images, dim=[0, 2, 3])

    return mean, std


def get_mean_and_std(dataloader, batch_size):

    channels_sum, channels_squared_sum, num_images = 0, 0, 0
    for data, _ in dataloader:

        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_images += 1

        if num_images == batch_size:
            break

    print("batch size 2: {}".format(num_images))
    mean = channels_sum / num_images

    # std = sqrt(E[X^2] - (E[X])^2)
    std = torch.sqrt((channels_squared_sum / num_images) - mean ** 2)

    return mean, std


def print_summary(model):

    summary(model, (3, 633, 600))


if __name__ == '__main__':
    args = args_parser()

    if not torch.cuda.is_available():
        print("GPU not available!!!!")

    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    global_model = EffNetB7()

    # print_summary(global_model)

    train_data = torchvision.datasets.ImageFolder(
        root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    # print("Loading data")
    # print(len(train_data))
    # print(train_data[0][0].size())

    _batch_size = 16

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=_batch_size,
                                              shuffle=False, num_workers=8)

    print("Calculating mean and std from dataset")
    mean, std = mean_std(trainloader)

    print(mean)
    print(std)

    print("Calculating mean and std from dataset again")
    mean, std = get_mean_and_std(trainloader, _batch_size)

    print(mean)
    print(std)

    sys.exit(0)

    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    print("Starting training...")
    for epoch in range(args.epochs):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            print("Calculating inferences...")
            outputs = global_model(images)

            print("Calculating loss...")
            loss = criterion(outputs, labels)

            print("Backpropagation...")
            loss.backward()

            print("Updating optimizer")
            optimizer.step()

            print('Train Epoch: {}, Batch: {}'.format(epoch+1, batch_idx))
            print("Loss: {}".format(loss.item()))

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('save/baseline_{}_{}_{}_loss.png'.format(args.dataset, args.model,
                                                         args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
