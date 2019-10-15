import torch
import torch.nn as nn


class SixtyChannels(nn.Module):
    def __init__(self):
        super(SixtyChannels, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=60, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)

        self.fc1 = nn.Linear(128*24*24, 128*128)
        self.fc2 = nn.Linear(128*128, 100*100)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        x = x.view(-1, 128*24*24)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    x = torch.rand(2, 60, 100, 100)
    net = SixtyChannels()
    output = net(x)
    print(output.shape)
