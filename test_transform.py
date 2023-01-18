import torchvision
from torchvision import transforms
import torch
import keep_aspect_ratio
import albumentations as A
import cv2
import albumentations.pytorch as album_pytorch
import numpy as np


TRAIN_DATA_PATH = "./original_dataset_rgba"

WIDTH = 384
HEIGHT = 380
AR_INPUT = WIDTH / HEIGHT

base_augmentations = [
    A.Resize(height=HEIGHT, width=WIDTH, interpolation=cv2.INTER_CUBIC),
    A.Flip(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    A.Perspective(p=1.0),
    A.Sharpen(p=1.0),
    album_pytorch.transforms.ToTensorV2()]


class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


train_data = torchvision.datasets.ImageFolder(
    root=TRAIN_DATA_PATH, transform=Transforms(transforms=A.Compose(base_augmentations)))

data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=20,
                                                shuffle=True)


count = 0
for batch_idx, (images, labels) in enumerate(data_loader_train):

    for img in images:
        transform = torchvision.transforms.ToPILImage()
        img = transform(img)
        img.save("augmented_data/" + str(count) + ".png", quality=100)
        count = count + 1

    # Iterate for 5 batches only (100 images)
    if batch_idx == 4:
        break
