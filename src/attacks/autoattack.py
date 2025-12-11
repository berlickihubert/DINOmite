"""
Wrapper for AutoAttack evaluation.

AutoAttack is an ensemble of attacks that provides a reliable evaluation.
Uses the official autoattack library if available, otherwise falls back to simplified version.

Based on: Croce & Hein, "Reliable evaluation of adversarial robustness with an ensemble of
diverse parameter-free attacks" (2020)
Paper: https://arxiv.org/abs/2003.01690
"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import official AutoAttack
try:
    from autoattack import AutoAttack

    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    logger.warning("AutoAttack library not available. Using simplified version.")
    from src.attacks import pgd_attack, fgsm_attack


def autoattack_evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float = 8 / 255,
    num_samples: Optional[int] = None,
) -> float:
    """
    AutoAttack evaluation using official library if available, otherwise simplified version.

    Full AutoAttack includes: APGD-CE, APGD-DLR, FAB, Square.
    If autoattack library is not available, uses simplified ensemble of PGD and FGSM.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
        eps: Attack strength (epsilon)
        num_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        Accuracy percentage (minimum across all attacks)
    """
    model.eval()

    if AUTOATTACK_AVAILABLE:
        # Use official AutoAttack
        try:
            adversary = AutoAttack(model, norm="Linf", eps=eps, version="standard", device=device)

            # Collect data
            x_test = []
            y_test = []
            for images, labels in dataloader:
                if num_samples and len(x_test) >= num_samples:
                    break
                x_test.append(images)
                y_test.append(labels)

            x_test = torch.cat(x_test, dim=0)[:num_samples] if num_samples else torch.cat(x_test, dim=0)
            y_test = torch.cat(y_test, dim=0)[:num_samples] if num_samples else torch.cat(y_test, dim=0)

            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # Run AutoAttack - run_standard_evaluation returns accuracy directly
            # Note: This method evaluates internally and returns accuracy
            accuracy = adversary.run_standard_evaluation(x_test, y_test, bs=dataloader.batch_size)
            return accuracy

        except Exception as e:
            logger.warning(f"AutoAttack library failed: {e}. Falling back to simplified version.")
            # Fall through to simplified version

    # Simplified version using ensemble of attacks
    attacks = [
        ("PGD", lambda m, img, lbl: pgd_attack(m, img, lbl, eps=eps, alpha=eps / 10, steps=40)),
        ("FGSM", lambda m, img, lbl: fgsm_attack(m, img, lbl, epsilon=eps)),
    ]

    results = []

    for attack_name, attack_fn in attacks:
        attack_correct = 0
        attack_total = 0

        for images, labels in dataloader:
            if num_samples and attack_total >= num_samples:
                break

            images = images.to(device)
            labels = labels.to(device)

            # Apply attack
            adv_images = attack_fn(model, images, labels)

            # Evaluate
            with torch.no_grad():
                outputs = model(adv_images)
                _, preds = torch.max(outputs, 1)
                attack_correct += (preds == labels).sum().item()
                attack_total += labels.size(0)

        accuracy = 100 * attack_correct / attack_total if attack_total > 0 else 0
        results.append(accuracy)
        logger.info(f"{attack_name} accuracy: {accuracy:.2f}%")

    # Return minimum accuracy (worst case)
    min_acc = min(results) if results else 0.0
    return min_acc
