#!/usr/bin/env python3
"""
Evaluate model robustness against a specific attack.

This script evaluates a trained model against a single attack method
and saves results for analysis.
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
from src.evaluation import evaluate_attack_robustness, evaluate_model
from src.config import RESULTS_DIR, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ATTACK_FUNCTIONS = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
    "bim": bim_attack,
    "deepfool": deepfool_attack,
    "cw": carlini_wagner_attack,
    "autoattack": autoattack_evaluate,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against attack")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=list(ATTACK_FUNCTIONS.keys()),
        help="Attack method to evaluate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "gtsrb", "tiny_imagenet"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0, 1/255, 2/255, 4/255, 8/255, 16/255],
        help="Epsilon values to test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
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
        pin_memory=True
    )

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = DinoWithLinearHead(num_classes=num_classes, model_path=args.model_path)
    model = model.to(device)
    model.eval()

    # Evaluate clean accuracy first
    logger.info("Evaluating clean accuracy...")
    clean_acc = evaluate_model(model, test_loader, device, attack_fn=None, desc="Clean")
    logger.info(f"Clean accuracy: {clean_acc:.2f}%")

    # Evaluate attack
    if args.attack == "autoattack":
        # AutoAttack has special handling
        logger.info("Running AutoAttack evaluation...")
        max_eps = args.epsilons[-1] if args.epsilons else 8/255
        acc = autoattack_evaluate(
            model, test_loader, device,
            eps=max_eps,
            num_samples=args.num_samples
        )
        results = {max_eps: {"accuracy": acc}}
    else:
        logger.info(f"Evaluating {args.attack.upper()} attack...")
        attack_fn = ATTACK_FUNCTIONS[args.attack]

        # Create wrapper function for evaluation
        def attack_wrapper(model, images, labels):
            if args.attack == "fgsm":
                return attack_fn(model, images, labels, labels, epsilon=args.epsilons[-1])
            elif args.attack == "pgd":
                return attack_fn(model, images, labels, eps=args.epsilons[-1], alpha=2/255, steps=40)
            elif args.attack == "bim":
                return attack_fn(model, images, labels, eps=args.epsilons[-1], alpha=2/255, steps=10)
            elif args.attack == "deepfool":
                return attack_fn(model, images, labels)
            elif args.attack == "cw":
                # C&W needs target labels
                target_labels = torch.randint(0, num_classes, labels.shape).to(device)
                target_labels = torch.where(target_labels == labels, (labels + 1) % num_classes, target_labels)
                return attack_fn(model, images, labels, target_labels, c=1.0, max_iter=100)
            else:
                return images

        # Evaluate with different epsilons
        results = {}
        for eps in args.epsilons:
            logger.info(f"Evaluating with epsilon={eps:.6f}...")

            def attack_fn_eps(model, images, labels):
            if args.attack == "fgsm":
                return attack_fn(model, images, labels, epsilon=eps)
                elif args.attack == "pgd":
                    return attack_fn(model, images, labels, eps=eps, alpha=eps/10, steps=40)
                elif args.attack == "bim":
                    return attack_fn(model, images, labels, eps=eps, alpha=eps/10, steps=10)
                else:
                    return attack_wrapper(model, images, labels)

            acc = evaluate_model(
                model, test_loader, device,
                attack_fn=attack_fn_eps if eps > 0 else None,
                desc=f"{args.attack.upper()} (ε={eps:.4f})"
            )
            results[eps] = {"accuracy": acc}
            logger.info(f"Accuracy at ε={eps:.6f}: {acc:.2f}%")

    # Prepare results
    output_results = {
        "model_path": args.model_path,
        "attack": args.attack,
        "dataset": args.dataset,
        "clean_accuracy": clean_acc,
        "attack_results": results,
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"{args.attack}_{args.dataset}_results.json"
    )
    with open(output_file, 'w') as f:
        json.dump(output_results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Attack: {args.attack.upper()}")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print("\nAttack Results:")
    for eps, res in results.items():
        print(f"  ε={eps:.6f}: {res['accuracy']:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()

