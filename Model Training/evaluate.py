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


def evaluate(CapsNet, valLoader, criterion, device, args):
    correct, total, total_loss,accuracy, valid_loss = 0, 0, 0., 0, 0
    CapsNet.eval()

    if args.mode == 'DR':
        for batch_i, (images, target) in enumerate(valLoader):
            target = torch.eye(10).index_select(dim=0, index=target)
            images, target = images.cuda(), target.cuda()

            caps_output = CapsNet(images)
            loss = criterion(caps_output[0], target, images, caps_output[1])

            preds = torch.norm(caps_output[0], dim=-1)  # Capsule lengths for class prediction
            preds = torch.argmax(preds, dim=1)
            correct += torch.sum(preds == torch.argmax(target, dim=1)).item()

            total += target.size(0)
            total_loss += loss.item()
        accuracy = 100. * correct / total
        valid_loss = total_loss / len(valLoader)
        #Store the trained Model
        #torch.save(CapsNet.state_dict(),args.path/'.efficientNet.pth')  # finalCIFER10_

    elif args.mode == 'EM':
        test_len = len(valLoader)
        for batch_idx, (images, target) in enumerate(valLoader):
            # target= torch.FloatTensor(targets)
            # target= target.type(torch.int)
            images, target = images.to(device), target.to(device)
            output, reconstruction = CapsNet(images, target)
            loss, spreadLoss, reconstructionLoss = criterion(output, target, images, reconstruction, r=1)

            acc = Accuracy(output, target)
            torch.cuda.empty_cache()

            accuracy += acc[0].item()
            valid_loss += loss.item()

        valid_loss /= test_len
        accuracy /= test_len


    elif args.mode == 'RR':
        test_len = len(valLoader)
        test_len = len(valLoader)
        with torch.no_grad():
            for data, target in valLoader:
                data, target = data.to(device), target.to(device)
                output = CapsNet(data, target)
                loss, _, _ = criterion(output[0], target, data, output[2], 1)
                acc = Accuracy(output[0], target)
                torch.cuda.empty_cache()
                accuracy += acc[0].item()
                valid_loss += loss.item()

        valid_loss /= test_len
        accuracy /= test_len

    return valid_loss, accuracy
