#!/usr/bin/env python3
"""
Train models with adversarial defense methods.

Supports:
- PGD Adversarial Training
- TRADES
- MART
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DinoWithLinearHead
from src.datasets import load_dataset, DatasetType
from src.defenses.defense_methods import (
    adversarial_training_step,
    trades_training_step,
    mart_training_step,
)
from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    MODELS_DIR,
    TRADES_BETA,
    MART_BETA,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch_defense(
    model, train_loader, optimizer, criterion, device,
    defense_type, eps, alpha, steps, beta=None
):
    """Train for one epoch with defense."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if defense_type == "pgd":
            loss = adversarial_training_step(
                model, images, labels, optimizer, criterion,
                attack_type='PGD', eps=eps, alpha=alpha, steps=steps
            )
        elif defense_type == "trades":
            loss, nat_loss, rob_loss = trades_training_step(
                model, images, labels, optimizer, criterion,
                beta=beta or TRADES_BETA, eps=eps, alpha=alpha, steps=steps
            )
        elif defense_type == "mart":
            loss = mart_training_step(
                model, images, labels, optimizer, criterion,
                beta=beta or MART_BETA, eps=eps, alpha=alpha, steps=steps
            )
        else:
            raise ValueError(f"Unknown defense type: {defense_type}")

        running_loss += loss

        # Calculate accuracy on natural examples
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train model with adversarial defense")
    parser.add_argument(
        "--defense",
        type=str,
        required=True,
        choices=["pgd", "trades", "mart"],
        help="Defense method to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "gtsrb", "tiny_imagenet"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8.0/255.0,
        help="Epsilon for adversarial training"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0/255.0,
        help="Alpha (step size) for PGD"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=7,
        help="Number of PGD steps"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta parameter for TRADES/MART (default: 6.0 for TRADES, 5.0 for MART)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=MODELS_DIR,
        help="Directory to save model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name for saved model (default: {defense}_cifar10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Set default beta if not provided
    if args.beta is None:
        if args.defense == "trades":
            args.beta = TRADES_BETA
        elif args.defense == "mart":
            args.beta = MART_BETA

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

    # Set default model name
    if args.model_name is None:
        args.model_name = f"{args.defense}_{args.dataset}"

    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    train_dataset = load_dataset(dataset_type, train=True, download=True)
    test_dataset = load_dataset(dataset_type, train=False, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = DinoWithLinearHead(num_classes=num_classes)
    model = model.to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Training loop
    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Starting {args.defense.upper()} adversarial training...")
    logger.info(f"Parameters: eps={args.eps:.4f}, alpha={args.alpha:.4f}, steps={args.steps}")
    if args.defense in ["trades", "mart"]:
        logger.info(f"Beta: {args.beta}")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch_defense(
            model, train_loader, optimizer, criterion, device,
            args.defense, args.eps, args.alpha, args.steps, args.beta
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        logger.info(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"{args.model_name}_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }, checkpoint_path)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(args.output_dir, f"{args.model_name}_best.pth")
            model.save(best_model_path)
            logger.info(f"Saved best model (acc: {best_acc:.2f}%) to {best_model_path}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, f"{args.model_name}_final.pth")
    model.save(final_model_path)
    logger.info(f"Training complete! Final model saved to {final_model_path}")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
