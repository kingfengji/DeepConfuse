import argparse
import torch
import torch.nn as nn
import os.path as osp
from models import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate adv data')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path for adv model')
    parser.add_argument('--eps', type=float, default=0.3, help='eps for data')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")

    atkmodel = Autoencoder().to(args.device)
    atkmodel.load_state_dict(torch.load(args.model_path))
    atkmodel.eval()
    print('load from ' + args.model_path)
    traindata, trainlabel = torch.load('data/MNIST/processed/training.pt')
    testdata, testlabel = torch.load('data/MNIST/processed/test.pt')
    traindata_adv = torch.zeros_like(traindata).float()
    testdata_adv = torch.zeros_like(testdata).float()

    for idx, i in enumerate(traindata):
        data_float = i.float()[None, None, :, :].to(args.device)/255.
        with torch.no_grad():
            noise = atkmodel(data_float) * args.eps
            atkdata = torch.clamp(data_float + noise, 0, 1)
        traindata_adv[idx] = atkdata[0,0].cpu()

    for idx, i in enumerate(testdata):
        data_float = i.float()[None, None, :, :].to(args.device)/255.
        with torch.no_grad():
            noise = atkmodel(data_float) * args.eps
            atkdata = torch.clamp(data_float + noise, 0, 1)
        testdata_adv[idx] = atkdata[0,0].cpu()

    torch.save(traindata_adv, 'training_adv.pt')
    torch.save(testdata_adv, 'test_adv.pt')