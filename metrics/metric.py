import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_accuracy(model, data):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in torch.utils.data.DataLoader(data, batch_size=len(data)):
            inputs, labels  = inputs.to(device), labels.to(device)
            output = model(inputs) # We don't need to run F.softmax
            # loss = criterion(inputs, labels)
            # val_loss += loss.item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inputs.shape[0]
        model.train()
        # average_loss = val_loss / len(dataloader)
    return correct / total
