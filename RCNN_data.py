import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data loading and batching
def load_mnist(batch_size, train=True):
    torch.set_num_threads(12)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]) 
    dataset = datasets.MNIST(
        root="../Data/mnist_data",
        train=train, 
        download=True,
        transform=transform
    ) 
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=1, pin_memory=True)
