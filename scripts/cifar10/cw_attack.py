import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linear_head_model import DinoWithLinearHead

print("Loading CIFAR-10 test dataset...")
dataset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transforms.ToTensor())

print("Initializing model...")
device = "cpu" #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DinoWithLinearHead('../../models/cifar10_linear_classifier.pth')
model = model.to(device)
model.eval()

def carlini_wagner_attack(model, img, label, c=1e-2, kappa=0, max_iter=1000, learning_rate=1e-2):
    w = torch.atanh((img * 1.999999) - 1).detach()
    w.requires_grad = True

    optimizer = optim.Adam([w], lr=learning_rate)

    target_label = (label + 1) % 10

    for step in range(max_iter):
        optimizer.zero_grad()
        adv_img = 0.5 * (torch.tanh(w) + 1)

        outputs = model(adv_img)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        real = logits.gather(1, label.unsqueeze(1))
        other = logits.gather(1, target_label.unsqueeze(1))
        print(f"real={real.item()}, other={other.item()}, diff={real.item() - other.item() + kappa}")

        f_loss = torch.clamp(real - other + kappa, min=0)
        l2dist = torch.norm(adv_img - img)
        loss = l2dist + c * f_loss

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
        print(f'Step {step}: Loss={loss.item():.4f}, f_loss={f_loss.item():.4f}, l2dist={l2dist.item():.4f}, Predicted={pred.item()}, Target={target_label.item()}')

    return adv_img.detach()

img, label = dataset[random.randint(0, len(dataset)-1)]
print(f"Original label: {label}")
out = model(img)
img = img.unsqueeze(0).to(device)
label = torch.tensor([label]).to(device)
adv_img = carlini_wagner_attack(model, img, label)
print(model(adv_img))