import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

dataset = MNIST(root="./data", download=True, transform=ToTensor())

val_size = 10000
batch_size = 64

train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
)

train_mean = 0.0
train_std = 0.0

for images, _ in train_loader:
    batch_samples = images.size(
        0
    )  # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)

    train_mean += images.mean(2).sum(0)
    train_std += images.std(2).sum(0)

train_mean /= len(train_loader.dataset)
train_std /= len(train_loader.dataset)

print('Mean: ', train_mean)
print('Std: ', train_std)

dataset = MNIST(
    root="./data",
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1308,), (0.3016,)),
        ]
    ),
)

test_dataset = MNIST(
    root="./data",
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1308,), (0.3016,))]
    ),
)

torch.manual_seed(1)

train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

print("Train: ", len(train_ds), "Val: ", len(val_ds), "Test: ", len(test_dataset))

train_loader = DataLoader(
    train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size * 2, num_workers=4, pin_memory=True
)