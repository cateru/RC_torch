import torch


def test(model, test_load, loss_fn, device):
    model.eval() 
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_load: 
            data, target = data.to(device), target.to(device) 
            output, _, _ = model(data) 
            test_loss += loss_fn(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_load)
    accuracy = 100. * correct / len(test_load.dataset) 
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_load.dataset)} ({accuracy:.2f}%)")