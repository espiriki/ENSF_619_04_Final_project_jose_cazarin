import torch
from torchvision import transforms
import keep_aspect_ratio
import torchvision
import math
import gc
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


class BaseBin():

    def __init__(self, args, path_to_dataset, width, height, batch_size, num_workers, name, device):

        self.name = name
        self.width = width
        self.height = height
        self.aspect_ratio = self.width/self.height
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.args = args
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        base_augmentations = [
            A.SafeRotate(p=1.0, interpolation=cv2.INTER_CUBIC,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0),
            keep_aspect_ratio.PadToMaintainAR(aspect_ratio=self.aspect_ratio),
            A.Resize(width=self.width,
                     height=self.height,
                     interpolation=cv2.INTER_CUBIC),
            A.Flip(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.Sharpen(p=1.0),
            A.Perspective(p=1.0, fit_output=True,
                          keep_size=True,
                          pad_mode=cv2.BORDER_CONSTANT,
                          pad_val=0)]

        black_bin_tranforms = base_augmentations.copy()
        green_bin_tranforms = base_augmentations.copy()
        blue_bin_tranforms = base_augmentations.copy()

        # Those per-channel mean and std values were obtained using the
        # calculate_mean_std_dataset.py script
        black_bin_tranforms.append(A.Normalize([0.5149, 0.4969, 0.4590],
                                               [0.3608, 0.3542, 0.3597]))

        blue_bin_tranforms.append(A.Normalize([0.5886, 0.5712, 0.5501],
                                              [0.3881, 0.3829, 0.3896]))

        green_bin_tranforms.append(A.Normalize([0.5768, 0.5347, 0.4923],
                                               [0.3913, 0.3880, 0.3957]))

        transforms_to_be_used = []
        if name == "Black bin":
            transforms_to_be_used = black_bin_tranforms
        elif name == "Green bin":
            transforms_to_be_used = green_bin_tranforms
        elif name == "Blue bin":
            transforms_to_be_used = blue_bin_tranforms

        transforms_to_be_used.append(a_pytorch.transforms.ToTensorV2())

        transforms_to_be_used = A.Compose(transforms_to_be_used)

        self.train_data = torchvision.datasets.ImageFolder(
            path_to_dataset, transform=Transforms(img_transf=transforms_to_be_used))

        self.data_loader = torch.utils.data.DataLoader(dataset=self.train_data,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)

    def local_update_weights(self, model):

        print("\n")
        print("Starting local update on {}".format(self.name))

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                     weight_decay=1e-4)
        model.to(self.device)

        num_batches = math.ceil(
            (len(self.data_loader.dataset)/self.batch_size))

        for local_epoch in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.data_loader):

                images, labels = images.to(self.device), labels.to(self.device)

                images = images.float()
                optimizer.zero_grad()
                model_outputs = model(images)
                loss = self.criterion(model_outputs, labels)
                loss.backward()
                optimizer.step()

                print("Batches {}/{} on local epoch {} out of {} local epochs".format(batch_idx,
                                                                                      num_batches, local_epoch, self.args.local_ep), end='\r')

            batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        model.to("cpu")
        self.criterion.to("cpu")

        print("\n")
        num_samples_dataset = len(self.data_loader.dataset)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), num_samples_dataset
