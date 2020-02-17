import os
import torch
import datetime
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import time
from models import *

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



def create_net():
    return VGG11()


def train(args, atkmodel, tgtmodel, clsmodel, device, train_loader,
          tgtoptimizer, clsoptimizer, epoch, clsepoch):
    st_time = time.time()
    atkmodel.eval()
    clsmodel.train()
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
        paragrads = torch.autograd.grad(
            loss, clsmodel.parameters(), create_graph=True)
        for i, (layername, layer) in enumerate(clsmodel.named_parameters()):
            modulenames, weightname = \
                layername.split('.')[:-1], layername.split('.')[-1]
            module = tmpmodel._modules[modulenames[0]]
            # TODO: could be potentially faster if we save the intermediate mappings
            for name in modulenames[1:]:
                module = module._modules[name]
            module._parameters[weightname] = \
                layer - clsoptimizer.param_groups[0]['lr'] * paragrads[i]
        tgtoptimizer.zero_grad()
        loss2 = -F.cross_entropy(tmpmodel(data), target)
        loss2.backward()
        tgtoptimizer.step()

        noise = atkmodel(data) * args.eps
        atkdata = torch.clamp(data + noise, 0, 1)
        output = clsmodel(atkdata)
        loss = F.cross_entropy(output, target)
        clsoptimizer.zero_grad()
        loss.backward()
        clsoptimizer.step()

        if batch_idx % args.log_interval == 0:
            with torch.no_grad():
                lossclean = F.cross_entropy(clsmodel(data), target)
            print('Train Epoch: {}, {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  '\tNatLoss:{:.6f}--->{:.6f}'.format(
                      epoch, clsepoch, batch_idx * len(data),
                      len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), loss.item(),
                      lossclean.item(),
                      -loss2.item()))
    atkloss = sum(losslist) / len(losslist)
    WRITER.add_scalar('train/loss(atk)', atkloss,
                      global_step=(epoch-1)*args.clsepochs+clsepoch)
    print('train time:', time.time() - st_time)
    return atkloss


def test(args, atkmodel, scratchmodel, device, train_loader, test_loader,
         epoch, trainepoch):
    st_time = time.time()
    test_loss = 0
    correct = 0
    atkmodel.eval()
    testoptimizer = optim.SGD(scratchmodel.parameters(), lr=args.lr)
    for i in range(trainepoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            testoptimizer.zero_grad()
            with torch.no_grad():
                noise = atkmodel(data) * args.eps
                atkdata = torch.clamp(data + noise, 0, 1)
            output = scratchmodel(atkdata)
            loss = F.cross_entropy(output, target)
            with torch.no_grad():
                output2 = scratchmodel(data)
                loss2 = F.cross_entropy(output2, target)
            loss.backward()
            testoptimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    'Test_train Epoch: {}, {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNatLoss: {:.6f}'.format(
                        epoch, i, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss.item(),
                        loss2.item()
                    ))

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
                 100. * correct/ len(test_loader.dataset)))

    accorig =  100. * correct / len(test_loader.dataset)
    WRITER.add_scalar('acc(origin)', accorig, global_step=epoch-1)
    print('test time:', time.time() - st_time)
    return accorig


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch MNIST Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs for atk to train')
    parser.add_argument('--clsepochs', type=int, default=20, metavar='N',
                        help='number of epochs for cls to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-atk', type=float, default=0.0001,
                        help='learning rate for attack model')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--test-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before '
                             'serious testing')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before '
                             'logging training status')
    parser.add_argument('--eps', type=float, default=0.032,
                        help='epsilon for data poisoning')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--path', type=str, default='', help='resume from checkpoint')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trainset = datasets.CIFAR10('data/cifar10', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))

    testset = datasets.CIFAR10('data/cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    atkmodel = UNet(3).to(args.device)
    if args.path:
        atkmodel.load_state_dict(torch.load(args.path))
    tgtmodel = UNet(3).to(args.device)
    tgtmodel.load_state_dict(atkmodel.state_dict())
    clsmodel = create_net().to(args.device)
    tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=args.lr_atk)
    best_acc = 100
    trainloss = 100
    for epoch in range(1, args.epochs + 1):
        clsoptimizer = optim.SGD(clsmodel.parameters(), lr=args.lr)
        for i in range(args.clsepochs):
            trainloss = train( #noqa
                args, atkmodel, tgtmodel, clsmodel, args.device, train_loader,
                tgtoptimizer, clsoptimizer, epoch, i)
        atkmodel.load_state_dict(tgtmodel.state_dict())
        clsmodel = create_net().to(args.device)
        scratchmodel = create_net().to(args.device)
        if epoch % args.test_interval == 0 or epoch == args.epochs:
            acc = test(args, atkmodel, scratchmodel, args.device,
                       train_loader, test_loader, epoch, trainepoch=args.clsepochs)
            if acc < best_acc:
                best_acc = acc
                torch.save(atkmodel.state_dict(), "atk.%03f.best.pth"%(args.eps))
            torch.save(atkmodel.state_dict(), "atk.%03f.latest.pth"%(args.eps))


if __name__ == '__main__':
    main()
