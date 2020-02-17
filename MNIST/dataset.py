from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import torch
import os.path as osp


class MNIST(data.Dataset):
    def __init__(self, root, train=True):
        if train:
            self.datas, self.labels = torch.load(osp.join(root, 'training.pt'))
        else:
            self.datas, self.labels = torch.load(osp.join(root, 'test.pt'))
        self.datas = self.datas[:, None, :, :].float() / 255.

    def __getitem__(self, index):
        img, target = self.datas[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.datas)
