import os
import time
import scipy
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split        
from ModelTraining.modeltraining import modelTraining


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--batchSize', type=int, default= 2, metavar='N',
                        help='input batch size for training (default: 128)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

parser.add_argument('--resultsFolder', type=str, default='/home/saja/PycharmProjects/pythonProject1/P1ModelInterpretation/Results/P1Project', metavar='SF',
                        help='where to store the snapshots()')

parser.add_argument('--data-folder', type=str, default='../data', metavar='DF',
                        help='where to store the datasets')


parser.add_argument('--dataset', type=str, default='MNIST', metavar='D',
                        help='dataset for training(MNIST, SVHN, CIFAR, CelebA)')

parser.add_argument('--mode', type=str, default='RR', metavar='D',
                        help='dataset for training(DR, EM, RR,ER)')

parser.add_argument('--preTrainedModel', type=str, default='/home/saja/PycharmProjects/pythonProject1/P1ModelInterpretation/BestModels/MNISTEM.pth', metavar='SF',
                        help='where to store the snapshots(path--)')

parser.add_argument('--task', type=str, default='training', metavar='N',
                        help='Train or interpret the models (training or interpretation)')

parser.add_argument('--classesNum', type=int, default=10, metavar='N',
                        help='number of classes (default: 10)')

parser.add_argument('--PCchannels', type=int, default= 6, metavar='N',
                        help='input batch size for training (default: 6 (28 input size), 8 (32 input size), 24 (64 input size))')

parser.add_argument('--inputSize', type=int, default= 28, metavar='N',
                        help='input batch size for training (default: 28, 32, and64)')

parser.add_argument('--nEpochs', type=int, default= 60, metavar='N',
                        help='number of epochs')

parser.add_argument('--nFilters', type=int, default= 128, metavar='N',
                        help='number of filters in the conv layer (default: 256, 128, 64 ..)')

parser.add_argument('--inputChannels', type=int, default= 1, metavar='N',
                        help='input batch size for training (default: 1 (grayScale),3(RGB))')

parser.add_argument('--numSamples', type=int, default= 0, metavar='Y',
                        help='number of samples (default: 1,51)')

'''---------Mostly EM and RR routing setting --------------------------'''
## the initial setting for EM
## CapsA, CapsB, CapsC, CapsD, FF, G, E = 32, 32, 32, 32, 16, 16, 5
#CapsA, CapsB, CapsC, CapsD= 32, 32, 32, 32

## the initial setting for RR
## CapsA, CapsB, CapsC, CapsD, FF, G, E = 32, 32, 32, 32, 16, 16, 5
parser.add_argument('--lr', type=float, default=3e-3, metavar='LR', #3e-3
                        help='learning rate (default: 0.01)')

parser.add_argument('--weight-decay', type=float, default=2e-7, metavar='WD',
                        help='weight decay (default: 0)')  # moein - according to openreview (2e-7)

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--iters', type=int, default=3, metavar='N',
                        help='iterations of EM Routing')

parser.add_argument('--CapsA', type=int, default=12, metavar='N',
                        help='defult 32')
parser.add_argument('--CapsB', type=int, default=12, metavar='N',
                        help='defult 32')
parser.add_argument('--CapsC', type=int, default=12, metavar='N',
                        help='defult 32')
parser.add_argument('--CapsD', type=int, default=12, metavar='N',
                        help='defult 32')


def get_setting(args):
    CapsNet=0
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'MNIST':
        im_transforms = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST("/home/saja/PycharmProjects/pythonProject1/P1ModelInterpretation/Datasets/MNISTData", train=True,
                                            download=True, transform=im_transforms)
        test_data = datasets.MNIST("/home/saja/PycharmProjects/pythonProject1/P1ModelInterpretation/Datasets/MNISTData", train=False,
                                           download=True, transform=im_transforms)

        testGT = datasets.MNIST("/home/saja/PycharmProjects/pythonProject1/P1ModelInterpretation/Datasets/MNISTData", train=False,
                                           download=True, transform=im_transforms)

        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batchSize, shuffle=False)
        train_, valid_ = random_split(train_data, [50000, 10000])

        
        validloader = DataLoader(valid_, batch_size=args.batchSize)
        trainloader = DataLoader(train_, batch_size=args.batchSize)

    
    return trainloader, validloader, testloader


'''Model Loadding..'''
args = parser.parse_args()
print(f'the args  are {args}')
args.cuda = not args.no_cuda and torch.cuda.is_available()
#torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
print(f'the device in  this case is {device}')

trainLoader, valLoader, testLoader= get_setting(args)

if args.task == 'training':
    modelTraining(args, device, trainLoader, valLoader, testLoader)

#elif args.task == 'interpretation':

