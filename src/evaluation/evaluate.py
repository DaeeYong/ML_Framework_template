import torch
import torch.nn as nn
from .metrics import accuracy

def evaluate(model, dataloader, device):
    """
    모델의 평가를 수행하고, 평균 loss와 accuracy를 반환.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            total_samples += batch_y.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc