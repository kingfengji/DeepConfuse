import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def create_net():
    net = nn.Sequential(
        nn.Conv2d(1, 20, 5, 1),
        nn.BatchNorm2d(20),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(20, 50, 5, 1),
        nn.BatchNorm2d(50),
        nn.ReLU(),
        nn.AvgPool2d(2),
        Flatten(),
        nn.Linear(4 * 4 * 50, 500),
        nn.BatchNorm1d(500),
        nn.ReLU(),
        nn.Linear(500, 10),
        nn.BatchNorm1d(10)
    )
    return net

def create_small_net():
    net = nn.Sequential(
        nn.Conv2d(1, 10, 5, 1),
        nn.BatchNorm2d(10),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(10, 25, 5, 1),
        nn.BatchNorm2d(25),
        nn.ReLU(),
        nn.AvgPool2d(2),
        Flatten(),
        nn.Linear(4 * 4 * 25, 250),
        nn.BatchNorm1d(250),
        nn.ReLU(),
        nn.Linear(250, 10),
        nn.BatchNorm1d(10)
    )
    return net

def create_large_net():
    net = nn.Sequential(
        nn.Conv2d(1, 40, 5, 1),
        nn.BatchNorm2d(40),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(40, 100, 5, 1),
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.AvgPool2d(2),
        Flatten(),
        nn.Linear(4 * 4 * 100, 1000),
        nn.BatchNorm1d(1000),
        nn.ReLU(),
        nn.Linear(1000, 10),
        nn.BatchNorm1d(10)
    )
    return net

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
