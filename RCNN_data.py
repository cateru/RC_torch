from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data loading and batching
def load_mnist(batch_size, train=True):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="mnist_data",
        train=train, 
        download=True,
        transform=transform
    ) 
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
