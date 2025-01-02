import time
import torch.optim as optim
from CapsuleNetworks.DynamicRouting.CapsuleLoss import CapsuleLoss
from CapsuleNetworks.DynamicRouting.dynamicRouting import CapsuleNetwork
from ModelTraining.train import train
from ModelTraining.evaluate import evaluate
from CapsuleNetworks.emRouting.network import *
from CapsuleNetworks.ResidualRouting.network import *
from CapsuleNetworks.emRouting.loss.capsuleLoss import CapsuleLoss

def modelTraining(args, device, trainLoader, valLoader, testLoader):
    t = time.time()
    tLoss = []
    vLoss = []
    tsLoss = []
    train_acc = []
    valid_acc = []
    test_acc = []

    ''' Network Loading'''
    if args.mode == 'DR':
        criterion = CapsuleLoss()
        CapsNet = CapsuleNetwork(args)
        CapsNet = CapsNet.to(device)
        optimizer = optim.Adam(CapsNet.parameters())
    elif args.mode == 'EM':
        CapsNet = capsules(args).to(device)
        criterion = CapsuleLoss(num_class=args.classesNum, m_min=0.2, m_max=0.9)
        optimizer = optim.Adam(CapsNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        CapsNet = CapsNet.to(device)
    elif args.mode == 'RR':
        CapsNet = capsulesRR(args).to(device)
        criterion = CapsuleLoss(num_class=args.classesNum, m_min=0.2, m_max=0.9)
        optimizer = optim.Adam(CapsNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)
        CapsNet = CapsNet.to(device)

    for epoch in range(args.nEpochs):
        train_loss, train_accuracy = train(args, CapsNet, trainLoader, optimizer, criterion, device)
        valid_loss, valid_accuracy = evaluate(CapsNet, valLoader, criterion, device, args)
        test_loss, test_accuracy = evaluate(CapsNet, testLoader, criterion, device, args)

        tLoss.append(train_loss)
        vLoss.append(valid_loss)
        tsLoss.append(test_loss)
        train_acc.append(train_accuracy)
        valid_acc.append(valid_accuracy)
        test_acc.append(test_accuracy)

    print('train loss ', tLoss)
    print('valid loss ', vLoss)
    print('test loss ', tsLoss)

    print('train acc ', train_acc)
    print('valid acc ', valid_acc)
    print('test acc ', test_acc)
