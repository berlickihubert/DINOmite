"""
DeepFool attack implementation.

Based on: Moosavi-Dezfooli et al., "DeepFool: a simple and accurate method to fool deep neural networks" (2016)
Paper: https://arxiv.org/abs/1511.04599

DeepFool finds the minimal perturbation needed to cross the decision boundary.
It's more efficient than C&W but typically produces larger perturbations.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def deepfool_attack_simple(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    overshoot: float = 0.02,
) -> torch.Tensor:
    """
    Simplified DeepFool attack for faster computation.

    DeepFool iteratively finds the minimum perturbation to cross
    the decision boundary by computing the distance to the boundary
    in the direction of the closest wrong class.

    Args:
        model: Target model to attack
        images: Input images tensor (B, C, H, W) in range [0, 1]
        labels: True labels tensor (B,)
        max_iter: Maximum iterations per image
        overshoot: Overshoot parameter to ensure misclassification

    Returns:
        Adversarial images tensor (B, C, H, W)

    References:
        Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016).
        DeepFool: a simple and accurate method to fool deep neural networks.
        arXiv preprint arXiv:1511.04599.
    """
    device = images.device
    was_single = False

    # Handle single image case
    if images.dim() == 3:
        images = images.unsqueeze(0)
        was_single = True
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    images = images.clone().detach()
    adv_images = images.clone()

    model.eval()

    for i in range(len(images)):
        image = images[i : i + 1].clone().detach()
        image.requires_grad = True

        # Get original prediction
        with torch.no_grad():
            orig_output = model(image)
            orig_pred = torch.argmax(orig_output, dim=1).item()

        for iteration in range(max_iter):
            image.requires_grad = True
            output = model(image)
            current_pred = torch.argmax(output, dim=1).item()

            # Check if attack succeeded
            if current_pred != orig_pred:
                break

            # Find the closest wrong class
            output_orig = output[0, orig_pred]
            sorted_indices = torch.argsort(output[0], descending=True)

            # Find first class that's not the original
            target_class = None
            for idx in sorted_indices:
                if idx.item() != orig_pred:
                    target_class = idx.item()
                    break

            if target_class is None:
                break

            output_target = output[0, target_class]

            # Compute gradient
            loss = output_target - output_orig
            grad = torch.autograd.grad(loss, image, retain_graph=False)[0]

            grad_norm = torch.norm(grad.view(-1))
            if grad_norm == 0:
                break

            # Compute perturbation (distance to decision boundary)
            r_i = (torch.abs(loss.item()) / (grad_norm**2)) * grad
            image = image + (1 + overshoot) * r_i
            image = image.detach()

        adv_images[i] = torch.clamp(image[0], 0, 1)

    # Return in original shape
    if was_single:
        return adv_images.squeeze(0).detach()
    return adv_images.detach()
