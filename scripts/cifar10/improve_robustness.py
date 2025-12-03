import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import os
import random
from tqdm import tqdm
from pgd_attack import pgd_attack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from linear_head_model import DinoWithLinearHead
from utils.save_example import save_images_side_by_side_with_logits

import torch.optim as optim

print("Loading CIFAR-10 training dataset...")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(project_root, "data")
os.makedirs(data_path, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root=data_path, train=True, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)

print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = os.path.join(project_root, "models", "cifar10_linear_classifier_robust.pth")
model = DinoWithLinearHead(model_path)
model = model.to(device)
model.train()


optimizer = optim.Adam(model.parameters(), lr=1e-5)

epochs = 50
steps_per_epoch = 100

print("Starting fine-tunining")

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for i, (images, labels) in enumerate(pbar):
        if i >= steps_per_epoch:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        model.eval() 
        adv_images = pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=7)
        model.train()
        
        optimizer.zero_grad()
        
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        
    
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (i + 1), 'acc': 100 * correct / total})
        
    print(f"Epoch {epoch+1} finished. Avg Loss: {running_loss / (i+1):.4f}, Accuracy: {100 * correct / total:.2f}%")
    save_path = os.path.join(project_root, "models", "cifar10_linear_classifier_robust.pth")
    torch.save({'model': model.state_dict()}, save_path)


print(f"Robust model saved to {save_path}")