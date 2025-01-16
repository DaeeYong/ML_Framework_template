import torch

def accuracy(outputs, targets):
    """
    outputs: (batch_size, num_classes)
    targets: (batch_size,)
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total