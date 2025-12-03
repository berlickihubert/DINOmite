import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import os
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from linear_head_model import DinoWithLinearHead
from utils.save_example import save_images_side_by_side_with_logits

print("Loading CIFAR-10 test dataset...")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_path = os.path.join(project_root, "data")
os.makedirs(data_path, exist_ok=True)

dataset = torchvision.datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=transforms.ToTensor()
)


print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DinoWithLinearHead("../../models/cifar10_linear_classifier.pth")
model = model.to(device)
model.eval()

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def pgd_attack(model, img, label, eps=8 / 255, alpha=2 / 255, steps=40):
    adv_img = img.clone().detach()
    original_img = img.clone().detach()

    for _ in range(steps):
        adv_img.requires_grad = True
        outputs = model(adv_img)
        loss = F.cross_entropy(outputs, label)
        model.zero_grad()
        loss.backward()
        grad_sign = adv_img.grad.sign()
        adv_img = adv_img.detach() + alpha * grad_sign
        eta = torch.clamp(adv_img - original_img, min=-eps, max=eps)
        adv_img = torch.clamp(original_img + eta, min=0, max=1)
    return adv_img.detach()


if __name__ == "__main__":
    attack_results = []
    for _ in tqdm(range(15), desc="Generating PGD examples"):
        img, label = dataset[random.randint(0, len(dataset) - 1)]
        img = img.unsqueeze(0).to(device)
        original_label = torch.tensor([label]).to(device)

        adv_img = pgd_attack(model, img, original_label)

        og_logits = model(img).detach().cpu().numpy().flatten()
        adv_logits = model(adv_img).detach().cpu().numpy().flatten()

        save_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../adversarial_examples")
        )
        os.makedirs(save_dir, exist_ok=True)

        i = 1
        while True:
            save_path = os.path.join(save_dir, f"pgd_example_cifar10_{i}.png")
            if not os.path.exists(save_path):
                break
            i += 1
        SAVE_PATH = save_path

        og_pred = int(og_logits.argmax())
        adv_pred = int(adv_logits.argmax())
        if og_pred != adv_pred:
            attack_results.append(f"{CIFAR10_CLASSES[og_pred]} -> {CIFAR10_CLASSES[adv_pred]}")
            save_images_side_by_side_with_logits(
                img.squeeze(0),
                adv_img.squeeze(0),
                og_logits,
                adv_logits,
                "Original (left) vs Adversarial PGD (right)",
                SAVE_PATH,
            )

    if attack_results:
        print("Successful attacks:")
        for result in attack_results:
            print(f"  - {result}")
    else:
        print("No attacks were successful in this run.")
    print(f"Examples saved in: {save_dir}")
