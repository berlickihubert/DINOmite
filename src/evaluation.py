"""
Evaluation utilities for adversarial robustness testing.

This module provides functions for evaluating model robustness against
various adversarial attacks and generating publication-ready results.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from tqdm import tqdm
from collections import defaultdict

from src.attacks import fgsm_attack, pgd_attack, bim_attack, deepfool_attack, carlini_wagner_attack


def evaluate_attack_robustness(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    attack_fn: Callable,
    attack_name: str,
    epsilons: Optional[List[float]] = None,
    num_samples: Optional[int] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model robustness against a specific attack.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
        attack_fn: Attack function to use
        attack_name: Name of the attack (for logging)
        epsilons: List of epsilon values to test (default: [0, 1/255, 2/255, 4/255, 8/255, 16/255])
        num_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        Dictionary mapping epsilon values to accuracy and attack success rate
    """
    model.eval()
    results = {}

    if epsilons is None:
        epsilons = [0, 1/255, 2/255, 4/255, 8/255, 16/255]

    if num_samples is None:
        num_samples = len(dataloader.dataset)

    for eps in epsilons:
        correct = 0
        total = 0
        successful_attacks = 0

        pbar = tqdm(dataloader, desc=f"{attack_name} (ε={eps:.4f})")
        for i, (images, labels) in enumerate(pbar):
            if total >= num_samples:
                break

            images = images.to(device)
            labels = labels.to(device)

            # Get original predictions
            with torch.no_grad():
                orig_outputs = model(images)
                orig_preds = torch.argmax(orig_outputs, dim=1)
                orig_correct = (orig_preds == labels).sum().item()

            # Apply attack
            if eps == 0:
                adv_images = images
            else:
                if attack_name == "FGSM":
                    target_labels = labels  # Untargeted
                    adv_images = attack_fn(model, images, labels, target_labels, epsilon=eps)
                elif attack_name == "PGD":
                    adv_images = attack_fn(model, images, labels, eps=eps, alpha=eps/10, steps=40)
                elif attack_name == "BIM":
                    adv_images = attack_fn(model, images, labels, eps=eps, alpha=eps/10, steps=10)
                elif attack_name == "DeepFool":
                    adv_images = attack_fn(model, images, labels)
                elif attack_name == "C&W":
                    # For C&W, we need target labels
                    target_labels = torch.randint(0, 10, labels.shape).to(device)
                    target_labels = torch.where(target_labels == labels,
                                                (labels + 1) % 10, target_labels)
                    adv_images = attack_fn(model, images, labels, target_labels,
                                         c=1.0, max_iter=100)
                else:
                    adv_images = images

            # Evaluate on adversarial examples
            with torch.no_grad():
                adv_outputs = model(adv_images)
                adv_preds = torch.argmax(adv_outputs, dim=1)
                adv_correct = (adv_preds == labels).sum().item()

            total += labels.size(0)
            correct += adv_correct
            successful_attacks += orig_correct - adv_correct

            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})

        accuracy = 100 * correct / total if total > 0 else 0
        attack_success_rate = 100 * successful_attacks / total if total > 0 else 0
        results[eps] = {
            'accuracy': accuracy,
            'attack_success_rate': attack_success_rate
        }

    return results


def generate_robustness_table(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Generate a comprehensive robustness table comparing different attacks.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
        num_samples: Maximum number of samples to evaluate

    Returns:
        Dictionary with results for each attack method
    """
    attacks = {
        'FGSM': lambda m, img, lbl: fgsm_attack(m, img, lbl, lbl, epsilon=8/255),
        'PGD': lambda m, img, lbl: pgd_attack(m, img, lbl, eps=8/255, alpha=2/255, steps=40),
        'BIM': lambda m, img, lbl: bim_attack(m, img, lbl, eps=8/255, alpha=2/255, steps=10),
    }

    results = {}

    # Clean accuracy
    model.eval()
    clean_correct = 0
    clean_total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            if clean_total >= num_samples:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            clean_correct += (preds == labels).sum().item()
            clean_total += labels.size(0)
    results['Clean'] = {'accuracy': 100 * clean_correct / clean_total}

    # Evaluate each attack
    for attack_name, attack_fn in attacks.items():
        correct = 0
        total = 0
        pbar = tqdm(dataloader, desc=f"Evaluating {attack_name}")
        for images, labels in pbar:
            if total >= num_samples:
                break
            images, labels = images.to(device), labels.to(device)
            adv_images = attack_fn(model, images, labels)
            with torch.no_grad():
                outputs = model(adv_images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})
        results[attack_name] = {'accuracy': 100 * correct / total if total > 0 else 0}

    return results


def plot_robustness_curves(
    results_dict: Dict[str, Dict[float, Dict[str, float]]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot robustness curves (accuracy vs epsilon) for different attacks.

    Args:
        results_dict: Dictionary mapping attack names to results (epsilon -> metrics)
        save_path: Path to save the plot (None = display)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for attack_name, results in results_dict.items():
        if attack_name == 'Clean':
            continue
        epsilons = sorted([k for k in results.keys() if isinstance(k, (int, float))])
        accuracies = [results[eps]['accuracy'] for eps in epsilons]
        ax.plot(epsilons, accuracies, marker='o', label=attack_name, linewidth=2)

    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Adversarial Robustness: Accuracy vs Epsilon', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved robustness curve to {save_path}")
    else:
        plt.show()
    plt.close()


def generate_results_table(
    results_dict: Dict[str, Dict[float, Dict[str, float]]],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a LaTeX-style table of results.

    Args:
        results_dict: Dictionary mapping attack names to results
        save_path: Path to save the table (None = don't save)

    Returns:
        DataFrame with results
    """
    # Prepare data for table
    data = []
    for attack_name, results in results_dict.items():
        if attack_name == 'Clean':
            data.append([attack_name, f"{results['accuracy']:.2f}", "-"])
        else:
            # Get accuracy at epsilon=8/255 if available
            eps_key = 8/255
            if eps_key in results:
                acc = results[eps_key]['accuracy']
                asr = results[eps_key]['attack_success_rate']
                data.append([attack_name, f"{acc:.2f}", f"{asr:.2f}"])
            else:
                # Get first non-zero epsilon
                eps_keys = [k for k in results.keys() if isinstance(k, (int, float)) and k > 0]
                if eps_keys:
                    eps_key = min(eps_keys)
                    acc = results[eps_key]['accuracy']
                    asr = results[eps_key]['attack_success_rate']
                    data.append([attack_name, f"{acc:.2f}", f"{asr:.2f}"])
                else:
                    data.append([attack_name, "N/A", "N/A"])

    df = pd.DataFrame(data, columns=['Attack Method', 'Accuracy (%)', 'Attack Success Rate (%)'])

    if save_path:
        # Save as CSV
        csv_path = save_path.replace('.tex', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved results table to {csv_path}")

        # Save as LaTeX
        tex_path = save_path if save_path.endswith('.tex') else save_path + '.tex'
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format="%.2f"))
        print(f"Saved LaTeX table to {tex_path}")

    return df


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    attack_fn: Optional[Callable] = None,
    desc: str = "Evaluating",
) -> float:
    """
    Evaluate model accuracy on clean or adversarial examples.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
        attack_fn: Optional attack function (None = clean evaluation)
        desc: Description for progress bar

    Returns:
        Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if attack_fn is not None:
            images = attack_fn(model, images, labels)

        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})

    return 100 * correct / total
