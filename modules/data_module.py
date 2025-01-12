import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import lightning as L


# AVAIL_GPUS = max(0, torch.cuda.device_count())


class MNISTDataModule(L.LightningDataModule):
    predict_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    def __init__(self, batch_size: int = 256) -> None:
        super().__init__()
        self.data_dir = "data"
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.BATCH_SIZE = batch_size

    def prepare_data(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        MNIST(self.data_dir, download=True)

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.BATCH_SIZE,
            num_workers=9,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.BATCH_SIZE,
            num_workers=9,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.BATCH_SIZE,
            num_workers=9,
            persistent_workers=True,
        )
