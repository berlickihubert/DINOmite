import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linear_head_model import DinoWithLinearHead

MODEL_FILE = '../../models/cifar10_linear_classifier.pth'
model = DinoWithLinearHead(MODEL_FILE)


def test_classifier():
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    print('🧪 Testing classifier...')
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'✅ Test Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return accuracy


if __name__ == '__main__':
    test_classifier()
