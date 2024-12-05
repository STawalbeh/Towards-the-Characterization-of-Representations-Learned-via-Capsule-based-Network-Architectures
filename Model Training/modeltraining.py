import time
import torch.optim as optim
from P1ModelInterpretation.Models.DynamicRouting.CapsuleLoss import CapsuleLoss
from P1ModelInterpretation.Models.DynamicRouting.dynamicRouting import CapsuleNetwork
from P1ModelInterpretation.ModelTraining.train import train
from P1ModelInterpretation.ModelTraining.evaluate import evaluate
from P1ModelInterpretation.Utilties.Visualization import trainingCurves


def modelTraining(args, device, trainLoader, valLoader, testLoader):
    t = time.time()
    tLoss = []
    vLoss = []
    tsLoss = []
    train_acc = []
    valid_acc = []
    test_acc = []

    ''' Network Loading'''
    criterion = CapsuleLoss()
    CapsNet = CapsuleNetwork(args)
    CapsNet = CapsNet.to(device)
    optimizer = optim.Adam(CapsNet.parameters())

    for epoch in range(args.nEpochs):
        train_loss, train_accuracy = train(CapsNet, trainLoader, optimizer, criterion, device)
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

    # Visualize the accuracy curves
    saveImgPath= 'YOUR path'
    label= 'Accuracy'
    title= 'TrainingAccuracies'
    trainingCurves(train_acc, valid_acc, test_acc, label, title, saveImgPath)
    # Visualize the loss curves
    label= 'Losses'
    title= 'TrainingLosses'
    trainingCurves(tLoss, vLoss, tsLoss, label, title, saveImgPath)

