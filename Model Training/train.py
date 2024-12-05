import torch


def train(CapsNet, trainLoader, optimizer, criterion, device):
    correct, total, total_loss = 0, 0, 0
    CapsNet.train()
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
    return train_loss, accuracy
