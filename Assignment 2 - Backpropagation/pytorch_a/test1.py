from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST training data with pytorch methods
transform=transforms.Compose([
    transforms.ToTensor(),
    ])
train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

dat = next(iter(train_loader))[0].numpy().reshape((60000,784))
output = next(iter(train_loader))[1].numpy()

print(train_dataset.targets[15000])
print(output[15000])