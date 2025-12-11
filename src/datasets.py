"""
Dataset loading and preprocessing utilities for DINOmite.
Supports multiple datasets with unified interface.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from typing import Optional
import logging
from pathlib import Path
import pandas as pd
from PIL import Image

from src.config import DatasetType, DATASET_CONFIGS, DATA_DIR

logger = logging.getLogger(__name__)


def get_cifar10_transforms(
    train: bool = True,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Get transforms for CIFAR-10 dataset.

    Args:
        train: Whether to use training transforms (with augmentation)
        normalize: Whether to normalize images

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
    ]

    if train:
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))

    if normalize:
        config = DATASET_CONFIGS[DatasetType.CIFAR10]
        transform_list.append(transforms.Normalize(mean=config.mean, std=config.std))

    return transforms.Compose(transform_list)


def get_gtsrb_transforms(
    train: bool = True,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Get transforms for GTSRB dataset.

    Args:
        train: Whether to use training transforms
        normalize: Whether to normalize images

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
    ]

    if train:
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))

    if normalize:
        config = DATASET_CONFIGS[DatasetType.GTSRB]
        transform_list.append(transforms.Normalize(mean=config.mean, std=config.std))

    return transforms.Compose(transform_list)


def get_tiny_imagenet_transforms(
    train: bool = True,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Get transforms for Tiny ImageNet dataset.

    Args:
        train: Whether to use training transforms
        normalize: Whether to normalize images

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
    ]

    if train:
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))

    if normalize:
        config = DATASET_CONFIGS[DatasetType.TINY_IMAGENET]
        transform_list.append(transforms.Normalize(mean=config.mean, std=config.std))

    return transforms.Compose(transform_list)


class GTSRBDataset(torch.utils.data.Dataset):
    """GTSRB dataset loader."""

    def __init__(self, root, train=True, transform=None, download=True):
        self.root = Path(root) / "GTSRB"
        self.train = train
        self.transform = transform

        if download:
            self._download()

        if train:
            annotations_file = self.root / "Train.csv"
            self.data_dir = self.root / "train"
        else:
            annotations_file = self.root / "Test.csv"
            self.data_dir = self.root / "test"

        if not annotations_file.exists():
            raise FileNotFoundError(
                f"GTSRB annotations not found at {annotations_file}. "
                "Please download GTSRB dataset manually from https://benchmark.ini.rub.de/gtsrb_dataset.html"
            )

        # Load annotations
        df = pd.read_csv(annotations_file)
        self.samples = []

        for _, row in df.iterrows():
            if train:
                img_path = self.data_dir / row["Path"]
            else:
                img_path = self.data_dir / row["Path"]

            if img_path.exists():
                self.samples.append((str(img_path), int(row["ClassId"])))

        logger.info(f"Loaded {len(self.samples)} GTSRB samples ({'train' if train else 'test'})")

    def _download(self):
        """Check if dataset exists, prompt user to download if not."""
        if not (self.root / "Train.csv").exists():
            logger.warning(
                f"GTSRB dataset not found at {self.root}. "
                "Please download from https://benchmark.ini.rub.de/gtsrb_dataset.html "
                f"and extract to {self.root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TinyImageNetDataset(torch.utils.data.Dataset):
    """Tiny ImageNet dataset loader."""

    def __init__(self, root, train=True, transform=None, download=True):
        self.root = Path(root) / "tiny-imagenet-200"
        self.train = train
        self.transform = transform

        if download:
            self._download()

        if not self.root.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet dataset not found at {self.root}. "
                "Please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                f"and extract to {self.root}"
            )

        if train:
            data_dir = self.root / "train"
        else:
            data_dir = self.root / "val"

        self.samples = []

        if train:
            # Training: images are in class folders
            for class_dir in sorted(data_dir.iterdir()):
                if class_dir.is_dir():
                    class_id = int(class_dir.name.split("_")[0]) if "_" in class_dir.name else class_dir.name
                    images_dir = class_dir / "images"
                    for img_file in sorted(images_dir.glob("*.JPEG")):
                        self.samples.append((str(img_file), class_id))
        else:
            # Validation: load from val_annotations.txt
            annotations_file = data_dir / "val_annotations.txt"
            if annotations_file.exists():
                with open(annotations_file, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        img_file = data_dir / "images" / parts[0]
                        class_name = parts[1]
                        # Get class ID from class name
                        class_id = self._get_class_id(class_name)
                        if img_file.exists():
                            self.samples.append((str(img_file), class_id))
            else:
                # Fallback: load from images directory
                images_dir = data_dir / "images"
                for img_file in sorted(images_dir.glob("*.JPEG")):
                    # Try to infer class from filename or use 0
                    self.samples.append((str(img_file), 0))

        logger.info(f"Loaded {len(self.samples)} Tiny ImageNet samples ({'train' if train else 'val'})")

    def _get_class_id(self, class_name):
        """Get class ID from class name."""
        # Get all class names
        classes_file = self.root / "words.txt"
        if classes_file.exists():
            with open(classes_file, "r") as f:
                class_names = [line.strip().split("\t")[0] for line in f]
                if class_name in class_names:
                    return class_names.index(class_name)
        return 0

    def _download(self):
        """Check if dataset exists, prompt user to download if not."""
        if not self.root.exists():
            logger.warning(
                f"Tiny ImageNet dataset not found at {self.root}. "
                "Please download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                f"and extract to {self.root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def _load_gtsrb(root, train, transform, download):
    """Load GTSRB dataset."""
    return GTSRBDataset(root=root, train=train, transform=transform, download=download)


def _load_tiny_imagenet(root, train, transform, download):
    """Load Tiny ImageNet dataset."""
    return TinyImageNetDataset(root=root, train=train, transform=transform, download=download)


def load_dataset(
    dataset_type: DatasetType,
    root: Optional[str] = None,
    train: bool = True,
    download: bool = True,
    normalize: bool = True,
) -> torch.utils.data.Dataset:
    """
    Load a dataset with appropriate transforms.

    Args:
        dataset_type: Type of dataset to load
        root: Root directory for dataset (default: PROJECT_ROOT/data)
        train: Whether to load training or test set
        download: Whether to download dataset if not present
        normalize: Whether to normalize images

    Returns:
        Dataset object

    Raises:
        ValueError: If dataset type is not supported
    """
    if root is None:
        root = DATA_DIR

    config = DATASET_CONFIGS[dataset_type]
    logger.info(f"Loading {config.name} dataset (train={train})")

    if dataset_type == DatasetType.CIFAR10:
        transform = get_cifar10_transforms(train=train, normalize=normalize)
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
    elif dataset_type == DatasetType.GTSRB:
        transform = get_gtsrb_transforms(train=train, normalize=normalize)
        dataset = _load_gtsrb(root=root, train=train, transform=transform, download=download)
    elif dataset_type == DatasetType.TINY_IMAGENET:
        transform = get_tiny_imagenet_transforms(train=train, normalize=normalize)
        dataset = _load_tiny_imagenet(root=root, train=train, transform=transform, download=download)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    logger.info(f"Loaded {len(dataset)} samples from {config.name}")
    return dataset


def get_dataset_classes(dataset_type: DatasetType) -> list:
    """
    Get class names for a dataset.

    Args:
        dataset_type: Type of dataset

    Returns:
        List of class names
    """
    if dataset_type == DatasetType.CIFAR10:
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset_type == DatasetType.GTSRB:
        # GTSRB has 43 classes - German traffic signs
        return [f"sign_{i}" for i in range(43)]
    elif dataset_type == DatasetType.TINY_IMAGENET:
        # Tiny ImageNet has 200 classes
        # Try to load actual class names if available
        try:
            from src.config import DATA_DIR

            words_file = Path(DATA_DIR) / "tiny-imagenet-200" / "words.txt"
            if words_file.exists():
                with open(words_file, "r") as f:
                    class_names = [line.strip().split("\t")[1] for line in f if "\t" in line]
                    if len(class_names) == 200:
                        return class_names
        except:
            pass
        return [f"class_{i}" for i in range(200)]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
