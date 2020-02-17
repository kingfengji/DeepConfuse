from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
from models import *


def clear_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

def train(args, atkmodel, tgtmodel, clsmodel, device, train_loader,
          tgtoptimizer, clsoptimizer, epoch):
    clsmodel.train()
    atkmodel.eval()
    tgtmodel.train()
    losslist = []
    for batch_idx, (data, target) in enumerate(train_loader):
        tmpmodel = create_net().to(device)
        data, target = data.to(device), target.to(device)
        noise = tgtmodel(data) * args.eps
        atkdata = torch.clamp(data + noise, 0, 1)
        output = clsmodel(atkdata)
        loss = F.cross_entropy(output, target)
        losslist.append(loss.item())
        clear_grad(clsmodel)
        paragrads = torch.autograd.grad(loss, clsmodel.parameters(),
                                        create_graph=True)
        for i, (layername, layer) in enumerate(clsmodel.named_parameters()):
            # layer = layer - args.lr * paragrads[i]
            modulename, weightname = layername.split('.')
            tmpmodel._modules[modulename]._parameters[
                weightname] = layer - args.lr * paragrads[i]
        tgtoptimizer.zero_grad()
        loss2 = -F.cross_entropy(tmpmodel(data), target)
        loss2.backward()
        tgtoptimizer.step()

        # for paracls, paratmp in zip(clsmodel.parameters(), tmpmodel.parameters()):
        #     paracls.data = paratmp.data
        noise = atkmodel(data) * args.eps
        atkdata = torch.clamp(data + noise, 0, 1)
        output = clsmodel(atkdata)
        loss = F.cross_entropy(output, target)
        clsoptimizer.zero_grad()
        loss.backward()
        clsoptimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  '\tNatLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                -loss2.item()))
    return sum(losslist) / len(losslist)


def test(args, atkmodel, scratchmodel, device, train_loader, test_loader, epoch,
         trainepoch):
    test_loss = 0
    correct = 0
    atkmodel.eval()
    testoptimizer = optim.SGD(scratchmodel.parameters(), lr=args.lr)
    for _ in range(trainepoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            testoptimizer.zero_grad()
            with torch.no_grad():
                noise = atkmodel(data) * args.eps
                atkdata = torch.clamp(data + noise, 0, 1)
                # atkdata = data + noise
            output = scratchmodel(atkdata)
            loss = F.cross_entropy(output, target)
            with torch.no_grad():
                output2 = scratchmodel(data)
                loss2 = F.cross_entropy(output2, target)
            loss.backward()
            testoptimizer.step()
            if batch_idx % args.log_interval * 100 == 0:
                print(
                    'Test_train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNatLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss.item(),
                        loss2.item()
                    ))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = scratchmodel(data)
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-atk', type=float, default=0.0001,
                        help='learning rate for attack model')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before '
                             'logging training status')
    parser.add_argument('--eps', type=float, default=0.3,
                        help='epsilon for data poisoning')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train-epoch', type=int, default=1, metavar='S',
                        help='training epochs for victim model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    atkmodel = Autoencoder().to(args.device)
    tgtmodel = Autoencoder().to(args.device)
    tgtmodel.load_state_dict(atkmodel.state_dict())
    clsmodel = create_net().to(args.device)
    tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=args.lr_atk)
    best_acc = 100
    trainloss = 100
    for epoch in range(1, args.epochs + 1):
        for i in range(args.train_epoch):
            clsoptimizer = optim.SGD(clsmodel.parameters(), lr=args.lr)
            trainloss = train(args, atkmodel, tgtmodel, clsmodel, args.device,
                              train_loader,
                              tgtoptimizer, clsoptimizer, epoch)
        atkmodel.load_state_dict(tgtmodel.state_dict())
        clsmodel = create_net().to(args.device)
        scratchmodel = create_net().to(args.device)
        acc = test(args, atkmodel, scratchmodel, args.device,
                   train_loader, test_loader, epoch, trainepoch=args.train_epoch)

        if (args.save_model):
            if acc < best_acc:
                best_acc = acc
                torch.save(atkmodel.state_dict(), "atkmnist_best.pt")
            torch.save(atkmodel.state_dict(), "atkmnist_latest.pt")


if __name__ == '__main__':
    main()
