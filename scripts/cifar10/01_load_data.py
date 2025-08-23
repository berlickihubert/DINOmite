import torch
import torchvision
import torchvision.transforms as transforms


# Add transformation to dataset to match DinoV3 ViT-S
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Download and load training data
train_dataset = torchvision.datasets.CIFAR10(
    root='../../data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='../../data', train=False, download=True, transform=transform
)

N = len(train_dataset) + len(test_dataset)
print(f'Train: {len(train_dataset)} samples, {100 * len(train_dataset) / N:.2f}%')
print(f'Test: {len(test_dataset)} samples, {100 * len(test_dataset) / N:.2f}%')
print(f'Classes: {train_dataset.classes}')

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
