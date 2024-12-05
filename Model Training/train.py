def train(capsule_net, train_loader, optimizer, criterion, device='cuda'):
    prediction= []
    batch=[]
    caps_output_all= []
    train_loss = 0
    train_loss2 =0
    correct, total, total_loss = 0, 0, 0
    
    capsule_net.train()
    for batch_i, (images, targets) in enumerate(train_loader):
        target = torch.eye(10).index_select(dim=0, index=targets)
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()
        caps_output = capsule_net(images)
        _, pred = torch.max(caps_output[0].data.cpu(), 1)
        loss = criterion(caps_output[0], target, images, caps_output[1])
        loss.backward()
        optimizer.step()
    
        preds = torch.norm(caps_output[0], dim=-1)  # Capsule lengths for class prediction
        preds= torch.argmax(preds, dim=1)
        correct += (preds == targets.to(device)).sum().item()  # Compare predictions to target
        total += target.size(0)
        total_loss += loss.item()
    accuracy = 100.*correct / total
    train_loss= total_loss/len(train_loader)
    return train_loss, accuracy
