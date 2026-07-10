import torch
from sklearn.metrics import confusion_matrix


def test(model, test_load, loss_fn, device):
    model.eval() 
    test_loss = 0
    correct = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_load: 
            data, target = data.to(device), target.to(device) 
            output, _, _ = model(data) 
            test_loss += loss_fn(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_loss /= len(test_load)
    accuracy = 100. * correct / len(test_load.dataset) 
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_load.dataset)} ({accuracy:.2f}%)")
    cm = confusion_matrix(all_targets, all_predictions)
    return accuracy, cm, all_targets, all_predictions

   