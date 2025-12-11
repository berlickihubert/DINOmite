"""
Fast Gradient Sign Method (FGSM) attack implementation.

Based on: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
Paper: https://arxiv.org/abs/1412.6572

FGSM is a single-step attack that moves in the direction of the gradient sign.
It's fast but less effective than iterative attacks like PGD.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8.0 / 255.0,
    targeted: bool = False,
    target_labels: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.

    FGSM computes the gradient of the loss with respect to the input
    and adds a small perturbation in the direction of the gradient sign.

    Args:
        model: Target model to attack
        images: Input images tensor (B, C, H, W) in range [0, 1]
        labels: True labels tensor (B,) for untargeted attack
        epsilon: Maximum perturbation (L∞ norm)
        targeted: Whether to perform targeted attack
        target_labels: Target labels tensor (B,) for targeted attack (required if targeted=True)

    Returns:
        Adversarial images tensor (B, C, H, W)

    References:
        Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).
        Explaining and harnessing adversarial examples.
        arXiv preprint arXiv:1412.6572.
    """
    device = images.device
    was_single = False

    # Handle single image case
    if images.dim() == 3:
        images = images.unsqueeze(0)
        was_single = True
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    # For targeted attack, use target_labels if provided, otherwise use labels
    if targeted:
        if target_labels is None:
            raise ValueError("target_labels must be provided for targeted attack")
        attack_labels = target_labels
    else:
        attack_labels = labels

    images = images.clone().detach().to(device)
    images.requires_grad = True

    model.eval()

    # Forward pass
    outputs = model(images)

    # Compute loss
    if targeted:
        loss = -F.cross_entropy(outputs, attack_labels)  # Negative for targeted
    else:
        loss = F.cross_entropy(outputs, attack_labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Generate adversarial examples
    with torch.no_grad():
        if targeted:
            adv_images = images - epsilon * images.grad.sign()
        else:
            adv_images = images + epsilon * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)

    # Return in original shape
    if was_single:
        return adv_images.squeeze(0).detach()
    return adv_images.detach()
