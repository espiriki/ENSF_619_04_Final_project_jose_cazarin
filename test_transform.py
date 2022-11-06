import torchvision
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
import torchvision.transforms as T
import keep_aspect_ratio
TRAIN_DATA_PATH = "./original_dataset_rgba"

WIDTH = 384
HEIGHT = 380
AR_INPUT = WIDTH / HEIGHT

TRANSFORM_IMG = transforms.Compose([
    keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
    transforms.Resize((WIDTH, HEIGHT), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

train_data = torchvision.datasets.ImageFolder(
    root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)

data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=20,
                                                shuffle=True)


count = 0
for batch_idx, (images, labels) in enumerate(data_loader_train):

    for img in images:
        transform = T.ToPILImage()
        img = transform(img)
        img.save("augmented_data/" + str(count) + ".png", quality=100)
        count = count + 1

    # Iterate for 5 batches only (100 images)
    if batch_idx == 4:
        break
