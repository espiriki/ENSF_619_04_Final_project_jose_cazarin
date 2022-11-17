import torch
from torchvision import transforms
import keep_aspect_ratio
import torchvision
import math
import gc


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

        TRANSFORM_IMG = transforms.Compose([
            transforms.RandomRotation(degrees=(-90, 90), expand=True),
            keep_aspect_ratio.PadToMaintainAR(aspect_ratio=self.aspect_ratio),
            transforms.Resize(
                (self.width, self.height), transforms.InterpolationMode.BICUBIC),
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

        self.train_data = torchvision.datasets.ImageFolder(
            root=path_to_dataset, transform=TRANSFORM_IMG)

        # self.train_data = torch.utils.data.Subset(self.train_data, range(10))

        self.data_loader = torch.utils.data.DataLoader(dataset=self.train_data,
                                                       batch_size=self.batch_size,
                                                       shuffle=True, num_workers=self.num_workers)

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

                optimizer.zero_grad()
                model_outputs = model(images)
                loss = self.criterion(model_outputs, labels)
                loss.backward()
                optimizer.step()

                print("Batches {}/{} on local epoch {} out of {} local epochs".format(batch_idx,
                                                                                      num_batches, local_epoch, self.args.local_ep), end='\r')
            batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        print("\n")
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
