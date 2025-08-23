import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm


# Data loading
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(
    root='../../data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='../../data', train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f'📊 Train: {len(train_dataset)} samples')
print(f'📊 Test: {len(test_dataset)} samples')

# Load DINOv3 model
print('🔄 Loading DINOv3 model...')
model = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')
print('✅ Model loaded successfully')

# Linear classifier
classifier = nn.Linear(384, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Training
print('Starting training...')
for epoch in range(10):
    epoch_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/10')
    for images, labels in progress_bar:
        # Convert tensor images to PIL for processor
        pil_images = [transforms.ToPILImage()(img) for img in images]
        inputs = processor(pil_images, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token

        logits = classifier(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = epoch_loss / num_batches
    print(f'Epoch {epoch + 1}/10 - Average Loss: {avg_loss:.4f}')

# Save trained classifier
print('\nSaving model...')
torch.save(classifier.state_dict(), '../../models/cifar10_linear_classifier.pth')
print('✅ Model saved to ../../models/cifar10_linear_classifier.pth')
