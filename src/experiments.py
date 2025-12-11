"""
Experiment configuration and management system.

This module provides utilities for managing experiments, configurations,
and result collection for adversarial robustness evaluation.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.config import PROJECT_ROOT


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    # Experiment metadata
    name: str
    description: str
    timestamp: str

    # Dataset configuration
    dataset: str = "cifar10"
    num_classes: int = 10

    # Model configuration
    model_type: str = "dinov3_linear"
    model_path: Optional[str] = None

    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 20
    optimizer: str = "adam"

    # Attack configuration (for evaluation)
    attack_type: Optional[str] = None
    attack_eps: float = 8.0 / 255.0
    attack_alpha: float = 2.0 / 255.0
    attack_steps: int = 40

    # Defense configuration (for training)
    defense_type: Optional[str] = None  # "pgd", "trades", "mart"
    defense_eps: float = 8.0 / 255.0
    defense_alpha: float = 2.0 / 255.0
    defense_steps: int = 7
    trades_beta: float = 6.0
    mart_beta: float = 5.0

    # Evaluation configuration
    eval_batch_size: int = 32
    num_eval_samples: Optional[int] = None  # None = all

    # Output paths
    output_dir: Optional[str] = None
    save_model: bool = True
    save_results: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**data)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ExperimentManager:
    """Manager for running and tracking experiments."""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize experiment manager.

        Args:
            base_dir: Base directory for experiments (default: PROJECT_ROOT/experiments)
        """
        if base_dir is None:
            base_dir = os.path.join(PROJECT_ROOT, "experiments")
        self.base_dir = base_dir
        self.configs_dir = os.path.join(base_dir, "configs")
        self.results_dir = os.path.join(base_dir, "results")
        self.logs_dir = os.path.join(base_dir, "logs")

        # Create directories
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        description: str = "",
        **kwargs,
    ) -> ExperimentConfig:
        """
        Create a new experiment configuration.

        Args:
            name: Experiment name
            description: Experiment description
            **kwargs: Additional configuration parameters

        Returns:
            ExperimentConfig object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{name}_{timestamp}"

        config = ExperimentConfig(
            name=exp_name,
            description=description,
            timestamp=timestamp,
            output_dir=os.path.join(self.results_dir, exp_name),
            **kwargs,
        )

        # Save config
        config_path = os.path.join(self.configs_dir, f"{exp_name}.json")
        config.save(config_path)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        return config

    def load_experiment(self, name: str) -> ExperimentConfig:
        """
        Load an experiment configuration.

        Args:
            name: Experiment name (with or without timestamp)

        Returns:
            ExperimentConfig object
        """
        # Try exact match first
        config_path = os.path.join(self.configs_dir, f"{name}.json")
        if os.path.exists(config_path):
            return ExperimentConfig.load(config_path)

        # Try to find by prefix
        configs = [f for f in os.listdir(self.configs_dir) if f.startswith(f"{name}_")]
        if configs:
            latest = sorted(configs)[-1]
            return ExperimentConfig.load(os.path.join(self.configs_dir, latest))

        raise FileNotFoundError(f"Experiment config not found: {name}")

    def list_experiments(self) -> List[str]:
        """List all experiment names."""
        configs = [f.replace(".json", "") for f in os.listdir(self.configs_dir) if f.endswith(".json")]
        return sorted(configs)

    def save_results(
        self,
        config: ExperimentConfig,
        results: Dict[str, Any],
        filename: str = "results.json",
    ) -> str:
        """
        Save experiment results.

        Args:
            config: Experiment configuration
            results: Results dictionary
            filename: Output filename

        Returns:
            Path to saved results file
        """
        output_path = os.path.join(config.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return output_path
