import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
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
device = "cpu"  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def carlini_wagner_attack(
    model, img, original_label, target_label, c=4e-2, kappa=0, max_iter=500, learning_rate=1e-2
):
    w = torch.atanh((img * 1.999999) - 1).detach()
    w.requires_grad = True

    optimizer = optim.Adam([w], lr=learning_rate)

    for _ in range(max_iter):
        optimizer.zero_grad()
        adv_img = 0.5 * (torch.tanh(w) + 1)

        outputs = model(adv_img)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        original_logit = logits.gather(1, original_label.unsqueeze(1))
        target_logit = logits.gather(1, target_label.unsqueeze(1))

        f_loss = torch.clamp(original_logit - target_logit + kappa, min=0)
        l2dist = torch.norm(adv_img - img)
        loss = l2dist + c * f_loss

        loss.backward()
        optimizer.step()

    return adv_img.detach()


attack_results = []
for _ in tqdm(range(15), desc="Generating C&W examples"):
    img, label = dataset[random.randint(0, len(dataset) - 1)]
    img = img.unsqueeze(0).to(device)
    original_label = torch.tensor([label]).to(device)
    target_class = random.choice([i for i in range(10) if i != label])
    target_label = torch.tensor([target_class]).to(device)
    adv_img = carlini_wagner_attack(model, img, original_label, target_label)

    og_logits = model(img).detach().cpu().numpy().flatten()
    adv_logits = model(adv_img).detach().cpu().numpy().flatten()

    save_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../adversarial_examples")
    )
    os.makedirs(save_dir, exist_ok=True)
    i = 1
    while True:
        save_path = os.path.join(save_dir, f"cw_example_cifar10_{i}.png")
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

print("Carlini-Wagner (C&W) attack examples generated.")
if attack_results:
    print("Successful attacks:")
    for result in attack_results:
        print(f"  - {result}")
print(
    f"Examples saved in: {os.path.abspath(os.path.join(os.path.dirname(__file__), '../../adversarial_examples'))}"
)
