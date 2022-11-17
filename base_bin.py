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

        base_augmentations = [
            transforms.RandomRotation(degrees=(-90, 90), expand=True),
            keep_aspect_ratio.PadToMaintainAR(aspect_ratio=self.aspect_ratio),
            transforms.Resize(
                (self.width, self.height), transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAutocontrast(),
            transforms.RandomPerspective(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor()]

        black_bin_tranforms = base_augmentations
        green_bin_tranforms = base_augmentations
        blue_bin_tranforms = base_augmentations

        # Those per-channel mean and std values were obtained using the
        # calculate_mean_std_dataset.py script
        black_bin_tranforms.append(transforms.Normalize([0.5149, 0.4969, 0.4590],
                                                        [0.3608, 0.3542, 0.3597]))
        blue_bin_tranforms.append(transforms.Normalize([0.5886, 0.5712, 0.5501],
                                                       [0.3881, 0.3829, 0.3896]))
        green_bin_tranforms.append(transforms.Normalize([0.5768, 0.5347, 0.4923],
                                                        [0.3913, 0.3880, 0.3957]))

        transforms_to_be_used = []
        if name == "Black bin":
            transforms_to_be_used = black_bin_tranforms
        elif name == "Green bin":
            transforms_to_be_used = green_bin_tranforms
        elif name == "Blue bin":
            transforms_to_be_used = blue_bin_tranforms

        print(transforms_to_be_used)

        self.train_data = torchvision.datasets.ImageFolder(
            root=path_to_dataset, transform=(transforms.Compose(transforms_to_be_used)))

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
