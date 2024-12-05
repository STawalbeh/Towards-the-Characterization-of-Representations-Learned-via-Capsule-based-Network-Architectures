def evaluate(capsule_net, valid_loader, optimizer, criterion):
    valid_loss = 0
    correct, total, total_loss = 0, 0, 0.
    capsule_net.eval()
    for batch_i, (images, target) in enumerate(valid_loader):
        target = torch.eye(10).index_select(dim=0, index=target)
        images, target = images.cuda(), target.cuda()
        
        caps_output = capsule_net(images)
        loss = criterion(caps_output[0], target, images, caps_output[1])
        
        preds = torch.norm(caps_output[0], dim=-1)  # Capsule lengths for class prediction
        preds= torch.argmax(preds, dim=1)
        correct += torch.sum(preds == torch.argmax(target, dim=1)).item()
        
        total += target.size(0)
        total_loss += loss.item()
    accuracy = 100.*correct / total
    valid_loss= total_loss/len(valid_loader)
    torch.save(capsule_net.state_dict(), './project_antwerp/extraFirstProject/PreTrainedCapsNets/efficientNet.pth') #finalCIFER10_
    return valid_loss, accuracy
