import torchvision
import torch
from subprocess import call

if __name__ == '__main__':
    torchvision.datasets.MNIST(root='data', download=True)