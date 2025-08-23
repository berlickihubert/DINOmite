import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Global model variables
MODEL_FILE = '../../models/checkpoint_epoch_3.pth'
model = None
processor = None
classifier = None


def load_model():
    global model, processor, classifier

    print('🔄 Loading DINOv3 model...')
    model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')

    classifier = nn.Linear(384, 10)

    # Load model weights
    if os.path.exists(MODEL_FILE):
        state_dict = torch.load(MODEL_FILE)
        # Check if it's a checkpoint or direct state dict
        if 'model' in state_dict:
            classifier.load_state_dict(state_dict['model'])
        else:
            classifier.load_state_dict(state_dict)
    else:
        # Load from checkpoint
        import glob

        checkpoints = glob.glob('../../models/linear_head_classification/checkpoint_epoch_*.pth')
        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint = torch.load(latest)
        classifier.load_state_dict(checkpoint['model'])

    classifier.eval()
    print('✅ Model loaded successfully')


def test_classifier():
    # Load test data
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
            pil_images = [transforms.ToPILImage()(img) for img in images]
            inputs = processor(pil_images, return_tensors='pt')

            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]

            logits = classifier(features)
            _, predicted = torch.max(logits, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'✅ Test Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return accuracy


if __name__ == '__main__':
    load_model()
    test_classifier()
