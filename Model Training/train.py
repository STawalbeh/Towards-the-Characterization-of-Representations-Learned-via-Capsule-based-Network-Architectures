import time
import torch

def Accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args, CapsNet, trainLoader, optimizer, criterion, device):
    CapsNet.train()
    train_loss, accuracy= 0, 0
    if args.mode == 'DR':
        correct, total, total_loss = 0, 0, 0
        for batch_i, (images, targets) in enumerate(trainLoader):
            target = torch.eye(10).index_select(dim=0, index=targets)
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            caps_output = CapsNet(images)
            _, pred = torch.max(caps_output[0].data.cpu(), 1)
            loss = criterion(caps_output[0], target, images, caps_output[1])
            loss.backward()
            optimizer.step()

            preds = torch.norm(caps_output[0], dim=-1)  # Capsule lengths for class prediction
            preds = torch.argmax(preds, dim=1)
            correct += (preds == targets.to(device)).sum().item()  # Compare predictions to target
            total += target.size(0)
            total_loss += loss.item()
        accuracy = 100. * correct / total
        train_loss = total_loss / len(trainLoader)

    elif args.mode == 'EM':
        torch.cuda.empty_cache()
        for batch_idx, (images, target) in enumerate(trainLoader):
            images, target = images.cuda(), target.cuda()
            optimizer.zero_grad()
            caps_output, reconstructions = CapsNet(images, target)
            loss, spreadLoss, reconstructionLoss = criterion(caps_output, target, images, reconstructions, 1)
            acc = Accuracy(caps_output, target)

            loss.backward()
            optimizer.step()
            accuracy += acc[0].item()
            train_loss += loss.item()

        accuracy /= len(trainLoader)
        train_loss /= len(trainLoader)

    elif args.mode == 'RR':
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_len = len(trainLoader)
        train_len = len(trainLoader)
        end = time.time()

        for batch_idx, (data, target) in enumerate(trainLoader):
            data_time.update(time.time() - end)
            data, target = data.to(device), target.to(device)
            CapsNet = CapsNet.to(device)
            optimizer.zero_grad()
            output = CapsNet(data, target)

            r = (1. * batch_idx + (args.nEpochs - 1) * train_len) / (args.nEpochs * train_len)
            loss, _, _ = criterion(output[0], target, data, output[2], 1)
            acc = Accuracy(output[0], target)
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            accuracy += acc[0].item()
            train_loss += loss.item()

        accuracy /= len(trainLoader)
        train_loss /= len(trainLoader)

    return train_loss, accuracy
