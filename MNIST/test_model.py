from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import *

def clear_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def test(args, atkmodel, scratchmodel, testoptimizer, device, train_loader,
         test_loader, epoch, labeltransform=None):
    test_loss = 0
    val_loss = 0
    correct = 0
    correct_val = 0
    correct_tgt = 0
    atkmodel.eval()
    targetlist = []
    predlist = []
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
                           100. * batch_idx / len(train_loader), loss.item(),
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
            if labeltransform is not None:
                correct_tgt += pred.eq(labeltransform(target)
                                       .view_as(pred)).sum().item()
            predlist = predlist + pred.cpu().numpy().reshape(-1).tolist()
            targetlist = targetlist + target.cpu().numpy().tolist()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            noise = atkmodel(data) * args.eps
            atkdata = torch.clamp(data + noise, 0, 1)
            output = scratchmodel(atkdata)
            val_loss += F.cross_entropy(output, target,
                                        reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1]  # get the index of the max log-probability
            correct_val += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.02f}%), '
        'Accuracy_tgt: {}/{} ({:.02f}%), Accuracy_adv: {}/{} ({:.02f}%)\n'
        .format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            correct_tgt, len(test_loader.dataset),
            100. * correct_tgt / len(test_loader.dataset),
            correct_val, len(test_loader.dataset),
            100. * correct_val / len(test_loader.dataset),
        ))
    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
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
    parser.add_argument('--path', type=str, required=True,
                        help='resume from checkpoint')
    parser.add_argument('--targeted', action='store_true', default=False)
    parser.add_argument('--modelsize', choices=['small', 'normal', 'large'],
                        help='choose different CNNs')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
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
    atkmodel.load_state_dict(torch.load(args.path))
    print('load from ' + args.path)
    if args.modelsize=='normal':
        scratchmodel = create_net().to(args.device)
    elif args.modelsize=='small':
        scratchmodel = create_small_net().to(args.device)
    elif args.modelsize=='large':
        scratchmodel = create_large_net().to(args.device)
    testoptimizer = optim.Adam(scratchmodel.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        acc = test(args, atkmodel, scratchmodel, testoptimizer, args.device,
                   train_loader, test_loader, epoch,
                   labeltransform=((lambda x: (x + 1) % 10)
                   if args.targeted else None))


if __name__ == '__main__':
    main()
