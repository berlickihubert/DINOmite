"""
Projected Gradient Descent (PGD) attack implementation.

Based on: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
Paper: https://arxiv.org/abs/1706.06083

PGD is an iterative attack that projects perturbations onto an epsilon ball.
It's considered one of the strongest white-box attacks.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 40,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.

    PGD performs multiple gradient steps, projecting the perturbation back
    onto the epsilon ball after each step. It starts from a random point
    within the epsilon ball for better exploration.

    Args:
        model: Target model to attack
        images: Input images tensor (B, C, H, W) in range [0, 1]
        labels: True labels tensor (B,)
        eps: Maximum perturbation (L∞ norm)
        alpha: Step size per iteration
        steps: Number of PGD iterations
        random_start: Whether to start from random point in epsilon ball

    Returns:
        Adversarial images tensor (B, C, H, W)

    References:
        Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
        Towards Deep Learning Models Resistant to Adversarial Attacks.
        arXiv preprint arXiv:1706.06083.
    """
    device = images.device
    was_single = False

    # Handle single image case
    if images.dim() == 3:
        images = images.unsqueeze(0)
        was_single = True
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    elif labels.dim() == 1 and labels.shape[0] != images.shape[0]:
        labels = labels.unsqueeze(0)

    adv_images = images.clone().detach()
    original_images = images.clone().detach()

    # Random initialization within epsilon ball
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0, 1)

    model.eval()

    for step in range(steps):
        adv_images.requires_grad = True

        # Forward pass
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update adversarial images
        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad_sign

        # Project back to epsilon ball
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        adv_images = torch.clamp(original_images + eta, min=0, max=1)

    # Return in original shape
    if was_single:
        return adv_images.squeeze(0).detach()
    return adv_images.detach()
