"""
Configuration and constants for DINOmite project.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import os


class DatasetType(Enum):
    """Supported dataset types."""

    CIFAR10 = "cifar10"
    GTSRB = "gtsrb"
    TINY_IMAGENET = "tiny_imagenet"


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    num_classes: int
    image_size: Tuple[int, int]  # (height, width)
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    download_url: Optional[str] = None
    paper_url: Optional[str] = None


# Dataset configurations
DATASET_CONFIGS: Dict[DatasetType, DatasetConfig] = {
    DatasetType.CIFAR10: DatasetConfig(
        name="CIFAR-10",
        num_classes=10,
        image_size=(224, 224),  # Resized for DINOv3
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        download_url="https://www.cs.toronto.edu/~kriz/cifar.html",
        paper_url="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf",
    ),
    DatasetType.GTSRB: DatasetConfig(
        name="GTSRB",
        num_classes=43,  # German Traffic Sign Recognition Benchmark
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        download_url="https://benchmark.ini.rub.de/gtsrb_news.html",
        paper_url="https://benchmark.ini.rub.de/gtsrb_dataset.html",
    ),
    DatasetType.TINY_IMAGENET: DatasetConfig(
        name="Tiny ImageNet",
        num_classes=200,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        download_url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        paper_url="http://cs231n.stanford.edu/project.html",
    ),
}

# Attack configurations
ATTACK_EPSILONS: List[float] = [0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]
DEFAULT_EPSILON: float = 8 / 255
DEFAULT_PGD_STEPS: int = 40
DEFAULT_PGD_ALPHA: float = 2 / 255

# Model configurations
DINO_MODEL_NAME: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINO_EMBEDDING_DIM: int = 384

# Training configurations
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_EPOCHS: int = 20

# Defense configurations
TRADES_BETA: float = 6.0
MART_BETA: float = 5.0

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ADVERSARIAL_EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "adversarial_examples")

# Paper references
PAPER_REFERENCES = {
    "madry_2018": {
        "title": "Towards Deep Learning Models Resistant to Adversarial Attacks",
        "authors": "Madry et al.",
        "year": 2018,
        "arxiv": "1706.06083",
        "url": "https://arxiv.org/abs/1706.06083",
    },
    "zhang_2019": {
        "title": "Theoretically Principled Trade-off between Robustness and Accuracy",
        "authors": "Zhang et al.",
        "year": 2019,
        "arxiv": "1901.08573",
        "url": "https://arxiv.org/abs/1901.08573",
    },
    "wang_2020": {
        "title": "Improving Adversarial Robustness Requires Revisiting Misclassified Examples",
        "authors": "Wang et al.",
        "year": 2020,
        "arxiv": None,
        "url": "https://openreview.net/pdf?id=rklOg6EFwS",
    },
    "moosavi_2016": {
        "title": "DeepFool: a simple and accurate method to fool deep neural networks",
        "authors": "Moosavi-Dezfooli et al.",
        "year": 2016,
        "arxiv": "1511.04599",
        "url": "https://arxiv.org/abs/1511.04599",
    },
    "kurakin_2017": {
        "title": "Adversarial examples in the physical world",
        "authors": "Kurakin et al.",
        "year": 2017,
        "arxiv": "1607.02533",
        "url": "https://arxiv.org/abs/1607.02533",
    },
    "carlini_2017": {
        "title": "Towards Evaluating the Robustness of Neural Networks",
        "authors": "Carlini & Wagner",
        "year": 2017,
        "arxiv": "1608.04644",
        "url": "https://arxiv.org/abs/1608.04644",
    },
    "goodfellow_2015": {
        "title": "Explaining and Harnessing Adversarial Examples",
        "authors": "Goodfellow et al.",
        "year": 2015,
        "arxiv": "1412.6572",
        "url": "https://arxiv.org/abs/1412.6572",
    },
    "croce_2020": {
        "title": "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks",
        "authors": "Croce & Hein",
        "year": 2020,
        "arxiv": "2003.01690",
        "url": "https://arxiv.org/abs/2003.01690",
    },
}
