#!/usr/bin/env python3
"""
Evaluate model robustness against all attacks.

This script runs comprehensive evaluation against all implemented attacks
and generates comparison plots and tables.
"""

import argparse
import torch
from torch.utils.data import DataLoader
import json
import os
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DinoWithLinearHead
from src.datasets import load_dataset, DatasetType
from src.attacks import (
    fgsm_attack,
    pgd_attack,
    bim_attack,
    deepfool_attack,
    carlini_wagner_attack,
    autoattack_evaluate,
)
from src.evaluation import evaluate_model, plot_robustness_curves, generate_results_table
from src.config import RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


ATTACKS = ["fgsm", "pgd", "bim", "deepfool", "cw", "autoattack"]


def evaluate_single_attack(model, test_loader, device, attack_name, epsilons):
    """Evaluate a single attack method."""
    logger.info(f"Evaluating {attack_name.upper()}...")

    results = {}

    # Clean accuracy
    if 0 in epsilons:
        clean_acc = evaluate_model(model, test_loader, device, attack_fn=None, desc="Clean")
        results[0] = {"accuracy": clean_acc}

    # Attack-specific evaluation
    if attack_name == "fgsm":
        for eps in epsilons:
            if eps == 0:
                continue

            def attack_fn(m, img, lbl):
                return fgsm_attack(m, img, lbl, epsilon=eps)

            acc = evaluate_model(model, test_loader, device, attack_fn=attack_fn, desc=f"FGSM (ε={eps:.4f})")
            results[eps] = {"accuracy": acc}

    elif attack_name == "pgd":
        for eps in epsilons:
            if eps == 0:
                continue

            def attack_fn(m, img, lbl):
                return pgd_attack(m, img, lbl, eps=eps, alpha=eps / 10, steps=40)

            acc = evaluate_model(model, test_loader, device, attack_fn=attack_fn, desc=f"PGD (ε={eps:.4f})")
            results[eps] = {"accuracy": acc}

    elif attack_name == "bim":
        for eps in epsilons:
            if eps == 0:
                continue

            def attack_fn(m, img, lbl):
                return bim_attack(m, img, lbl, eps=eps, alpha=eps / 10, steps=10)

            acc = evaluate_model(model, test_loader, device, attack_fn=attack_fn, desc=f"BIM (ε={eps:.4f})")
            results[eps] = {"accuracy": acc}

    elif attack_name == "deepfool":
        # DeepFool doesn't use epsilon in the same way
        def attack_fn(m, img, lbl):
            return deepfool_attack(m, img, lbl)

        acc = evaluate_model(model, test_loader, device, attack_fn=attack_fn, desc="DeepFool")
        results[8 / 255] = {"accuracy": acc}  # Use default epsilon for plotting

    elif attack_name == "cw":
        # C&W doesn't use epsilon in the same way
        num_classes = 10

        def attack_fn(m, img, lbl):
            target_labels = torch.randint(0, num_classes, lbl.shape).to(device)
            target_labels = torch.where(target_labels == lbl, (lbl + 1) % num_classes, target_labels)
            return carlini_wagner_attack(m, img, lbl, target_labels, c=1.0, max_iter=100)

        acc = evaluate_model(model, test_loader, device, attack_fn=attack_fn, desc="C&W")
        results[8 / 255] = {"accuracy": acc}  # Use default epsilon for plotting

    elif attack_name == "autoattack":
        # AutoAttack evaluation (returns single accuracy value)
        max_eps = max(epsilons) if epsilons else 8 / 255
        acc = autoattack_evaluate(model, test_loader, device, eps=max_eps)
        results[max_eps] = {"accuracy": acc}

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against all attacks")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "gtsrb", "tiny_imagenet"], help="Dataset to use"
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255],
        help="Epsilon values to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=ATTACKS,
        choices=ATTACKS,
        help="Attacks to evaluate",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Map dataset string to DatasetType
    dataset_map = {
        "cifar10": DatasetType.CIFAR10,
        "gtsrb": DatasetType.GTSRB,
        "tiny_imagenet": DatasetType.TINY_IMAGENET,
    }
    dataset_type = dataset_map[args.dataset]

    # Get number of classes from config
    from src.config import DATASET_CONFIGS

    num_classes = DATASET_CONFIGS[dataset_type].num_classes

    # Load dataset
    logger.info(f"Loading {args.dataset} test dataset...")
    test_dataset = load_dataset(dataset_type, train=False, download=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = DinoWithLinearHead(num_classes=num_classes, model_path=args.model_path)
    model = model.to(device)
    model.eval()

    # Evaluate all attacks
    all_results = {}

    for attack_name in args.attacks:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating {attack_name.upper()}")
        logger.info(f"{'=' * 60}")

        results = evaluate_single_attack(model, test_loader, device, attack_name, args.epsilons)
        all_results[attack_name] = results

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "all_attacks_results.json")

    output_data = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    # Generate plots
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_path = os.path.join(plots_dir, "robustness_curves.png")
    plot_robustness_curves(all_results, save_path=plot_path)
    logger.info(f"Plot saved to {plot_path}")

    # Generate table
    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    table_path = os.path.join(tables_dir, "robustness_table.tex")
    df = generate_results_table(all_results, save_path=table_path)
    logger.info(f"Table saved to {table_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print("\nResults:")
    for attack_name, results in all_results.items():
        print(f"\n{attack_name.upper()}:")
        for eps, res in sorted(results.items()):
            print(f"  ε={eps:.6f}: {res['accuracy']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
