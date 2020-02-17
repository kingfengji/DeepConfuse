import os
import torch
import datetime
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms, models
import torch.utils.data as data
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


cfg = {
    'CNNsmall': [32//2, 'M', 64//2, 'M', 128//2, 'M', 128//2, 'M', 128//2, 'M'],
    'CNN': [32, 'M', 64, 'M', 128, 'M', 128, 'M', 128, 'M'],
    'CNNlarge': [32*2, 'M', 64*2, 'M', 128*2, 'M', 128*2, 'M', 128*2, 'M']
}


def createvgg(cfg):
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=7, stride=1)]
    layers += [Flatten(), nn.Linear(128//2, 2)]
    return nn.Sequential(*layers)

class SimpleDataset(data.Dataset):
    def __init__(self, path, poisonratio=0):
        data, poisondata, target = torch.load(path)
        if poisonratio==0:
            self.datas = data
        else:
            self.datas = data
            poisonlen = int(poisondata.shape[0] * poisonratio)
            self.datas[:poisonlen] = poisondata[:poisonlen]
        self.labels = target

    def __getitem__(self, index):
        img, target = self.datas[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.datas)

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

def test(args, scratchmodel, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = scratchmodel(data)
            # sum up batch loss
            test_loss += F.cross_entropy(
                output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'. \
          format(test_loss, correct, len(test_loader.dataset),
                 100. * correct / len(test_loader.dataset)))

    accorig = 100. * correct / len(test_loader.dataset)
    return accorig


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch MNIST Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--clsepochs', type=int, default=5, metavar='N',
                        help='number of epochs for cls to train')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--poisonratio', type=float, default=0, metavar='M',
                        help='poison ratio, 0 means clean data')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--test-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before '
                             'serious testing')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before '
                             'logging training status')
    parser.add_argument('--arch', choices=['small', 'normal', 'large', 'resnet', 'densenet'], default='normal',
                        help='choose different CNNs')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    log = []
    args.device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_set = SimpleDataset('train_bin.pt', poisonratio=args.poisonratio)
    test_set = SimpleDataset('test_bin.pt', poisonratio=0)
    vali_set = SimpleDataset('val_bin.pt', poisonratio=args.poisonratio)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        vali_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    if args.arch == 'normal':
        clsmodel = createvgg(cfg['CNN'])
    elif args.arch == 'small':
        clsmodel = createvgg(cfg['CNNsmall'])
    elif args.arch == 'large':
        clsmodel = createvgg(cfg['CNNlarge'])
    elif args.arch == 'resnet':
        clsmodel = models.resnet50(pretrained=False)
        clsmodel.classifier = nn.Linear(clsmodel.fc.in_features, 2)
    elif args.arch == 'densenet':
        clsmodel = models.densenet121(pretrained=False)
        clsmodel.classifier = nn.Linear(clsmodel.classifier.in_features, 2)

    clsmodel = clsmodel.to(args.device)
    for i in range(args.clsepochs):
        lr = 0.01
        clsoptimizer = optim.SGD(clsmodel.parameters(), lr=args.lr)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            clsoptimizer.zero_grad()
            output = clsmodel(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            clsoptimizer.step()
        trainacc = test(args, clsmodel, args.device,
                            train_loader)
        valiacc = test(args, clsmodel, args.device,
                           val_loader)
        testacc = test(args, clsmodel, args.device,
                           test_loader)

if __name__ == '__main__':
    main()
