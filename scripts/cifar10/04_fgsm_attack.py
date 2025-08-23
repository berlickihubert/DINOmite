import torch
from model import model, dataloader


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)


# Test FGSM
epsilon = 0.1
image, label = next(iter(dataloader))
image.requires_grad = True

# Forward pass
output = model(image)
loss = torch.norm(output)  # Simple loss for feature perturbation

# Backward pass
model.zero_grad()
loss.backward()

# Generate adversarial example
perturbed_image = fgsm_attack(image, epsilon, image.grad.data)

# Compare outputs
with torch.no_grad():
    original_features = model(image)
    adversarial_features = model(perturbed_image)

    perturbation_norm = torch.norm(perturbed_image - image)
    feature_diff = torch.norm(adversarial_features - original_features)

    print(f'Perturbation L2 norm: {perturbation_norm:.4f}')
    print(f'Feature difference: {feature_diff:.4f}')
    print('FGSM attack completed')
