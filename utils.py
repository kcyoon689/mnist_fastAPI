import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import plotly.express as px
import plotly.graph_objects as go
import os


def get_dataloader(batch_size, val_size):
    dataset = MNIST(root="./data", download=True, transform=ToTensor())

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

    # print('Mean: ', train_mean)
    # print('Std: ', train_std)

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

    # print("Train: ", len(train_ds), "Val: ", len(val_ds), "Test: ", len(test_dataset))

    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size * 2, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# plot_losses_and_accuracy(history)
def plotly_plot_losses(train_loss, val_loss):
    train_losses = train_loss
    val_losses = val_loss

    fig = go.Figure()

    # Training losses
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(train_losses) + 1)),
            y=train_losses,
            mode="lines+markers",
            name="Training",
        )
    )

    # Validation losses
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(val_losses) + 1)),
            y=val_losses,
            mode="lines+markers",
            name="Validation",
        )
    )

    fig.update_layout(
        title="Loss vs. No. of epochs", xaxis_title="Epoch", yaxis_title="Loss"
    )

    save_path = "./results/"
    makedirs(save_path)
    fig.write_image(os.path.join(save_path, "loss_graph.png"))


def plotly_plot_scores(val_acc):
    # acc = [x['val_acc'] for x in history]
    acc = val_acc

    fig = go.Figure()

    # Validation accuracy
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(acc) + 1)),
            y=acc,
            mode="lines+markers",
            name="Validation Accuracy",
        )
    )

    fig.update_layout(
        title="Accuracy vs. No. of epochs", xaxis_title="Epoch", yaxis_title="Accuracy"
    )

    # fig.show()
    save_path = "./results/"
    makedirs(save_path)
    fig.write_image(os.path.join(save_path, "accuracy_graph.png"))


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloader(64, 10000)

    for images, labels in train_loader:
        # print(images[0].shape, labels[0])
        img = images[0].squeeze().numpy()
        fig = px.imshow(img, color_continuous_scale="gray")
        fig.show()
        break
