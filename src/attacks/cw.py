"""
Carlini-Wagner (C&W) L2 attack implementation.

Based on: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017)
Paper: https://arxiv.org/abs/1608.04644

C&W is an optimization-based attack that finds minimal L2 perturbations.
It's one of the strongest attacks but computationally expensive.
"""
import torch
import torch.optim as optim
from typing import Union
import logging

logger = logging.getLogger(__name__)


def carlini_wagner_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    original_labels: torch.Tensor,
    target_labels: torch.Tensor,
    c: float = 4e-2,
    kappa: float = 0.0,
    max_iter: int = 500,
    learning_rate: float = 1e-2,
) -> torch.Tensor:
    """
    Carlini-Wagner L2 attack.

    C&W uses an optimization approach to find minimal perturbations
    that cause misclassification. It uses a tanh transformation to
    ensure valid pixel values.

    Args:
        model: Target model to attack
        images: Input images tensor (B, C, H, W) in range [0, 1]
        original_labels: True labels tensor (B,)
        target_labels: Target labels for misclassification (B,)
        c: Trade-off parameter between perturbation size and attack success
        kappa: Confidence parameter (higher = more confident misclassification)
        max_iter: Maximum optimization iterations
        learning_rate: Learning rate for optimization

    Returns:
        Adversarial images tensor (B, C, H, W)

    References:
        Carlini, N., & Wagner, D. (2017).
        Towards evaluating the robustness of neural networks.
        arXiv preprint arXiv:1608.04644.
    """
    device = images.device
    was_single = False

    # Handle single image case
    if images.dim() == 3:
        images = images.unsqueeze(0)
        was_single = True
    if original_labels.dim() == 0:
        original_labels = original_labels.unsqueeze(0)
    if target_labels.dim() == 0:
        target_labels = target_labels.unsqueeze(0)

    batch_size = images.shape[0]
    adv_images = []

    model.eval()

    for i in range(batch_size):
        img = images[i:i+1]
        orig_label = original_labels[i:i+1]
        targ_label = target_labels[i:i+1]

        # Transform to tanh space for optimization
        w = torch.atanh((img * 1.999999) - 1).detach()
        w.requires_grad = True

        optimizer = optim.Adam([w], lr=learning_rate)

        for _ in range(max_iter):
            optimizer.zero_grad()

            # Transform back to image space
            adv_img = 0.5 * (torch.tanh(w) + 1)

            # Forward pass
            outputs = model(adv_img)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Compute loss components
            original_logit = logits.gather(1, orig_label.unsqueeze(1))
            target_logit = logits.gather(1, targ_label.unsqueeze(1))

            # Attack success term
            f_loss = torch.clamp(original_logit - target_logit + kappa, min=0)

            # Perturbation size term (L2 distance)
            l2dist = torch.norm(adv_img - img)

            # Combined loss
            loss = l2dist + c * f_loss

            loss.backward()
            optimizer.step()

        # Final adversarial image
        adv_img = 0.5 * (torch.tanh(w) + 1)
        adv_images.append(adv_img.detach())

    adv_images = torch.cat(adv_images, dim=0)

    # Return in original shape
    if was_single:
        return adv_images.squeeze(0)
    return adv_images
