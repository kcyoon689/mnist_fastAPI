import torch
import torch.nn as nn
import torch.nn.functional as F
from .trainer import MnistModelBase
from torchsummary import summary


class MnistModel(MnistModelBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3
            ),  # RF - 3x3 # 26x26
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 16, 3),  # RF - 5x5 # 24x24
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, 3),  # RF - 7x7 # 22x22
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
        )

        # translation layer
        # input - 22x22x64; output - 11x11x32
        self.trans1 = nn.Sequential(
            # RF - 7x7
            nn.Conv2d(32, 20, 1),  # 22x22
            nn.ReLU(),
            nn.BatchNorm2d(20),
            # RF - 14x14
            nn.MaxPool2d(2, 2),  # 11x11
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1),  # RF - 16x16 #output- 9x9
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout2d(0.1),
            nn.Conv2d(20, 16, 3),  # RF - 16x16 #output- 9x9
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 16, 3),  # RF - 18x18 #output- 7x7
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3),  # RF - 20x20  #output- 5x5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1),
            # nn.Conv2d(16,10,1),   #RF - 20x20  #output- 7x7
        )

        # GAP Layer
        self.avg_pool = nn.Sequential(
            # # RF - 22x22
            nn.AvgPool2d(5)
        )  ## output_size=1

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 10, 1),  # RF - 20x20  #output- 7x7
        )

    def forward(self, xb):
        x = self.conv1(xb)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg_pool(x)
        x = self.conv4(x)

        x = x.view(-1, 10)
        return x


# Print Summary of the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MnistModel().to(device)
    summary(model, input_size=(1, 28, 28))
    print(model)
