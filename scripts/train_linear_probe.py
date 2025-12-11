#!/usr/bin/env python3
"""
Train a linear probe classifier on top of DINOv3 features.

This script trains a linear classification head on frozen DINOv3 features.
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
from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    MODELS_DIR,
    PROJECT_ROOT,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
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
    parser = argparse.ArgumentParser(description="Train linear probe on DINOv3")
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
        "--output-dir",
        type=str,
        default=MODELS_DIR,
        help="Directory to save model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cifar10_linear_classifier",
        help="Name for saved model"
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

    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

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

