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
dataset = torchvision.datasets.CIFAR10(
    root="../../data", train=False, download=True, transform=transforms.ToTensor()
)

print("Initializing model...")
device = "cpu"
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


def fgsm_attack(model, img, original_label, target_label, epsilon=8 / 255):
    img_adv = img.clone().detach().to(device)
    img_adv.requires_grad = True
    outputs = model(img_adv)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    loss = F.cross_entropy(logits, target_label)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        adv_img = torch.clamp(img - epsilon * img_adv.grad.sign(), 0, 1)
    return adv_img.detach()


attack_results = []
for _ in tqdm(range(15), desc="Generating FGSM examples"):
    img, label = dataset[random.randint(0, len(dataset) - 1)]
    img = img.unsqueeze(0).to(device)
    original_label = torch.tensor([label]).to(device)
    target_class = random.choice([i for i in range(10) if i != label])
    target_label = torch.tensor([target_class]).to(device)
    adv_img = fgsm_attack(model, img, original_label, target_label)

    outputs_og = model(img)
    og_logits = (
        (outputs_og.logits if hasattr(outputs_og, "logits") else outputs_og)
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )
    outputs_adv = model(adv_img)
    adv_logits = (
        (outputs_adv.logits if hasattr(outputs_adv, "logits") else outputs_adv)
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )

    save_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../adversarial_examples")
    )
    os.makedirs(save_dir, exist_ok=True)
    i = 1
    while True:
        save_path = os.path.join(save_dir, f"fgsm_example_cifar10_{i}.png")
        if not os.path.exists(save_path):
            break
        i += 1
    SAVE_PATH = save_path
    save_dir = os.path.dirname(SAVE_PATH)
    os.makedirs(save_dir, exist_ok=True)

    og_pred = int(og_logits.argmax())
    adv_pred = int(adv_logits.argmax())
    if og_pred != adv_pred:
        attack_results.append(f"{CIFAR10_CLASSES[og_pred]} -> {CIFAR10_CLASSES[adv_pred]}")
        save_images_side_by_side_with_logits(
            img.squeeze(0),
            adv_img.squeeze(0),
            og_logits,
            adv_logits,
            "Original (left) vs Adversarial (right)",
            SAVE_PATH,
        )

print("\nFGSM attack examples generated.")
if attack_results:
    print("Successful attacks:")
    for result in attack_results:
        print(f"  - {result}")
print(
    f"\nExamples saved in: {os.path.abspath(os.path.join(os.path.dirname(__file__), '../../adversarial_examples'))}"
)
