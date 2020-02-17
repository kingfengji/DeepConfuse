import os
import torch
import datetime
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import time

LOGDIR = 'log/{progname}'.format(progname=__file__[:-3])
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
WRITER = SummaryWriter(LOGDIR)


def clear_grad(m):
    for p in m.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.shape[0], -1)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch MNIST Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='epsilon for data poisoning')
    parser.add_argument('--path', type=str, default='',
                        help='resume from checkpoint')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    log = []
    args.device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_set = datasets.ImageFolder('data/train',
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))
    val_set = datasets.ImageFolder('data/val',
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()
                                     ]))
    vali_set = datasets.ImageFolder('data/vali',
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        vali_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    atkmodel = Autoencoder().to(args.device)
    if args.path:
        atkmodel.load_state_dict(torch.load(args.path))
    cleandata = []
    cleantarget = []
    poisondata = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            noise = atkmodel(data) * args.eps
            atkdata = torch.clamp(data + noise, 0, 1)
        cleandata.append(data.to(torch.device('cpu')))
        cleantarget.append(target.to(torch.device('cpu')))
        poisondata.append(atkdata.to(torch.device('cpu')))
    cleandata = torch.cat(cleandata, dim=0)
    poisondata = torch.cat(poisondata, dim=0)
    cleantarget = torch.cat(cleantarget)
    torch.save((cleandata,poisondata,cleantarget), open('train_bin.pt', 'wb'))
    grids = []
    for i in range(2):
        grids += [cleandata[i * 4:i * 4 + 4], poisondata[i * 4:i * 4 + 4]]
    image = torchvision.utils.make_grid(torch.cat(grids),
                                        nrow=4,
                                        padding=2)
    torchvision.utils.save_image(image, 'imagenet.png')

    cleandata = []
    cleantarget = []
    poisondata = []
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            noise = atkmodel(data) * args.eps
            atkdata = torch.clamp(data + noise, 0, 1)
        cleandata.append(data.to(torch.device('cpu')))
        cleantarget.append(target.to(torch.device('cpu')))
        poisondata.append(atkdata.to(torch.device('cpu')))
    cleandata = torch.cat(cleandata, dim=0)
    poisondata = torch.cat(poisondata, dim=0)
    cleantarget = torch.cat(cleantarget)
    torch.save((cleandata, poisondata, cleantarget), open('val_bin.pt', 'wb'))

    cleandata = []
    cleantarget = []
    poisondata = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            noise = atkmodel(data) * args.eps
            atkdata = torch.clamp(data + noise, 0, 1)
        cleandata.append(data.to(torch.device('cpu')))
        cleantarget.append(target.to(torch.device('cpu')))
        poisondata.append(atkdata.to(torch.device('cpu')))
    cleandata = torch.cat(cleandata, dim=0)
    poisondata = torch.cat(poisondata, dim=0)
    cleantarget = torch.cat(cleantarget)
    torch.save((cleandata, poisondata, cleantarget), open('test_bin.pt', 'wb'))

if __name__ == '__main__':
    main()
