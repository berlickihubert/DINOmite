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

print("Loading CIFAR-10 test dataset...")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(project_root, "data")
os.makedirs(data_path, exist_ok=True)

test_dataset = torchvision.datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(model, dataloader, device, attack_fn=None, desc="Evaluating"):
    model.eval()
    correct = 0
    total = 0
    
    max_batches = 10 
    
    pbar = tqdm(dataloader, desc=desc, total=max_batches)
    for i, (images, labels) in enumerate(pbar):
        if i >= max_batches:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        if attack_fn:
            images = attack_fn(model, images, labels)
            
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        pbar.set_postfix({'acc': 100 * correct / total})
        
    return 100 * correct / total

print("\nEvaluating original model")
original_model_path = os.path.join(project_root, "models", "cifar10_linear_classifier.pth")
original_model = DinoWithLinearHead(original_model_path)
original_model = original_model.to(device)

acc_clean_orig = evaluate_model(original_model, test_loader, device, desc="Original Clean")
acc_pgd_orig = evaluate_model(original_model, test_loader, device, attack_fn=pgd_attack, desc="Original PGD")

print(f"Original model - general accuracy: {acc_clean_orig:.2f}%")
print(f"Original model - PGD accuracy: {acc_pgd_orig:.2f}%")

print("\Evaluating robust model")
robust_model_path = os.path.join(project_root, "models", "cifar10_linear_classifier_robust.pth")

if os.path.exists(robust_model_path):
    robust_model = DinoWithLinearHead(robust_model_path)
    robust_model = robust_model.to(device)

    acc_clean_robust = evaluate_model(robust_model, test_loader, device, desc="Robust Clean")
    acc_pgd_robust = evaluate_model(robust_model, test_loader, device, attack_fn=pgd_attack, desc="Robust PGD")

    print(f"Robust Model - general accuracy: {acc_clean_robust:.2f}%")
    print(f"Robust Model - PGD accuracy: {acc_pgd_robust:.2f}%")
    
else:
    print(f"Robust model not found at {robust_model_path}.")
