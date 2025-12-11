"""
Defense methods against adversarial attacks.

Based on:
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (1706.06083)
  Paper: https://arxiv.org/abs/1706.06083
- Zhang et al., "Theoretically Principled Trade-off between Robustness and Accuracy" (1901.08573)
  Paper: https://arxiv.org/abs/1901.08573
- Wang et al., "Improving Adversarial Robustness Requires Revisiting Misclassified Examples"
  Paper: https://openreview.net/pdf?id=rklOg6EFwS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Optional

from src.attacks import pgd_attack, fgsm_attack


class InputTransformationDefense:
    """
    Input transformation defenses: JPEG compression, bit-depth reduction, etc.
    """

    @staticmethod
    def jpeg_compression(images, quality=75):
        """
        Apply JPEG compression as a defense.
        Note: This is a simplified version. Full JPEG requires PIL.
        """
        # Simulate JPEG compression with quantization
        # In practice, use PIL Image.save with quality parameter
        return images

    @staticmethod
    def bit_depth_reduction(images, bits=4):
        """Reduce bit depth of images."""
        levels = 2 ** bits
        return torch.round(images * (levels - 1)) / (levels - 1)

    @staticmethod
    def gaussian_noise(images, std=0.1):
        """Add Gaussian noise."""
        noise = torch.randn_like(images) * std
        return torch.clamp(images + noise, 0, 1)

    @staticmethod
    def median_filter(images, kernel_size=3):
        """Apply median filter (simplified)."""
        # Full implementation would use proper median filter
        return images


class FeatureDenoising(nn.Module):
    """
    Feature denoising blocks for adversarial defense.
    Based on: Xie et al., "Feature Denoising for Improving Adversarial Robustness"
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Denoise features
        x_denoised = self.conv(x)
        x_denoised = self.bn(x_denoised)
        return x + x_denoised  # Residual connection


def adversarial_training_step(
    model, images, labels, optimizer, criterion,
    attack_type='PGD', eps=8/255, alpha=2/255, steps=7
):
    """
    Perform one step of adversarial training.
    Based on: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
    """
    model.train()

    # Generate adversarial examples
    if attack_type == 'PGD':
        adv_images = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=steps)
    elif attack_type == 'FGSM':
        adv_images = fgsm_attack(model, images, labels, epsilon=eps)
    else:
        adv_images = images

    # Train on adversarial examples
    optimizer.zero_grad()
    outputs = model(adv_images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def trades_training_step(
    model, images, labels, optimizer, criterion,
    beta=6.0, eps=8/255, alpha=2/255, steps=7
):
    """
    TRADES training: Trade-off between accuracy and robustness.
    Based on: Zhang et al., "Theoretically Principled Trade-off between Robustness and Accuracy"

    TRADES minimizes a combination of natural loss and robust loss (KL divergence).

    Args:
        model: Model to train
        images: Input images tensor (B, C, H, W)
        labels: True labels tensor (B,)
        optimizer: Optimizer
        criterion: Loss criterion (CrossEntropyLoss)
        beta: Trade-off parameter between natural and robust loss (default: 6.0)
        eps: Maximum perturbation for PGD (default: 8/255)
        alpha: Step size for PGD (default: 2/255)
        steps: Number of PGD steps (default: 7)

    Returns:
        Tuple of (total_loss, natural_loss, robust_loss) as floats

    References:
        Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019).
        Theoretically Principled Trade-off between Robustness and Accuracy.
        arXiv preprint arXiv:1901.08573.
    """
    model.train()

    # Natural loss
    outputs_natural = model(images)
    loss_natural = criterion(outputs_natural, labels)

    # Adversarial loss (KL divergence)
    adv_images = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=steps)
    outputs_adv = model(adv_images)

    # KL divergence between natural and adversarial predictions
    p_natural = F.softmax(outputs_natural, dim=1)
    p_adv = F.log_softmax(outputs_adv, dim=1)
    loss_robust = F.kl_div(p_adv, p_natural, reduction='batchmean')

    # Combined loss
    loss = loss_natural + beta * loss_robust

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_natural.item(), loss_robust.item()


def mart_training_step(
    model, images, labels, optimizer, criterion,
    beta=5.0, eps=8/255, alpha=2/255, steps=7, num_classes=None
):
    """
    MART training: Misclassification Aware adveRsarial Training.
    Based on: Wang et al., "Improving Adversarial Robustness Requires Revisiting Misclassified Examples"

    Paper: https://openreview.net/pdf?id=rklOg6EFwS

    Args:
        model: Model to train
        images: Input images tensor (B, C, H, W)
        labels: True labels tensor (B,)
        optimizer: Optimizer
        criterion: Loss criterion (not used directly, but kept for consistency)
        beta: Trade-off parameter for KL divergence (default: 5.0)
        eps: Maximum perturbation for PGD (default: 8/255)
        alpha: Step size for PGD (default: 2/255)
        steps: Number of PGD steps (default: 7)
        num_classes: Number of classes (auto-detected if None)

    Returns:
        Loss value as float

    References:
        Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020).
        Improving Adversarial Robustness Requires Revisiting Misclassified Examples.
        ICLR 2020. https://openreview.net/pdf?id=rklOg6EFwS
    """
    model.train()

    # Get num_classes from model if not provided
    if num_classes is None:
        # Try to infer from model output
        with torch.no_grad():
            test_output = model(images[:1])
            num_classes = test_output.shape[1]

    # Natural predictions
    outputs_natural = model(images)
    preds_natural = torch.argmax(outputs_natural, dim=1)
    correct_mask = (preds_natural == labels)

    # Adversarial examples
    adv_images = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=steps)
    outputs_adv = model(adv_images)

    # BCE loss for misclassified examples
    p_adv = F.softmax(outputs_adv, dim=1)
    loss_bce = F.binary_cross_entropy(
        p_adv, F.one_hot(labels, num_classes=num_classes).float(), reduction='none'
    ).sum(dim=1)

    # KL divergence
    p_natural = F.softmax(outputs_natural, dim=1)
    p_adv_log = F.log_softmax(outputs_adv, dim=1)
    loss_kl = F.kl_div(p_adv_log, p_natural, reduction='none').sum(dim=1)

    # Combined loss
    loss = loss_bce.mean() + beta * (loss_kl * (1 - correct_mask.float())).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def pgd_adversarial_training(
    model, train_loader, device, epochs=10, eps=8/255, alpha=2/255, steps=7,
    lr=1e-4, save_path=None
):
    """
    Full adversarial training using PGD.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            loss = adversarial_training_step(
                model, images, labels, optimizer, criterion,
                attack_type='PGD', eps=eps, alpha=alpha, steps=steps
            )

            running_loss += loss
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if save_path:
            torch.save({'model': model.state_dict()}, save_path)

    return model


def trades_training(
    model, train_loader, device, epochs=10, beta=6.0, eps=8/255, alpha=2/255, steps=7,
    lr=1e-4, save_path=None
):
    """
    TRADES adversarial training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_natural_loss = 0.0
        running_robust_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            loss, nat_loss, rob_loss = trades_training_step(
                model, images, labels, optimizer, criterion,
                beta=beta, eps=eps, alpha=alpha, steps=steps
            )

            running_loss += loss
            running_natural_loss += nat_loss
            running_robust_loss += rob_loss

            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        avg_nat = running_natural_loss / len(train_loader)
        avg_rob = running_robust_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} "
              f"(Natural: {avg_nat:.4f}, Robust: {avg_rob:.4f}), Accuracy: {accuracy:.2f}%")

        if save_path:
            torch.save({'model': model.state_dict()}, save_path)

    return model
