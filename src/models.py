"""
Model definitions and utilities for DINOmite.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from typing import Optional, Union
import os
import logging

from src.config import DINO_MODEL_NAME, DINO_EMBEDDING_DIM, PROJECT_ROOT, MODELS_DIR

logger = logging.getLogger(__name__)


class DinoWithLinearHead(nn.Module):
    """
    DINOv3 model with a linear classification head.

    This model uses a pretrained DINOv3 vision transformer as a feature extractor
    and adds a trainable linear layer for classification.

    Attributes:
        dino: Pretrained DINOv3 model
        processor: Image processor for DINOv3
        head: Linear classification head
    """

    def __init__(
        self,
        num_classes: int = 10,
        model_path: Optional[str] = None,
        dino_model_name: str = DINO_MODEL_NAME,
    ):
        """
        Initialize DINOv3 model with linear head.

        Args:
            num_classes: Number of output classes
            model_path: Path to saved model weights (optional)
            dino_model_name: Name of DINOv3 model to load
        """
        super().__init__()

        self.dino_model_name = dino_model_name
        self.num_classes = num_classes

        # Load DINOv3 backbone
        logger.info(f"Loading DINOv3 model: {dino_model_name}")
        self.dino = AutoModel.from_pretrained(dino_model_name)
        self.processor = AutoImageProcessor.from_pretrained(dino_model_name)

        # Freeze DINOv3 backbone (only train linear head)
        for param in self.dino.parameters():
            param.requires_grad = False

        # Linear classification head
        self.head = nn.Linear(DINO_EMBEDDING_DIM, num_classes)

        # Load weights if provided
        if model_path is not None:
            self.load_weights(model_path)

    def load_weights(self, model_path: str) -> None:
        """
        Load model weights from file.

        Args:
            model_path: Path to model weights file
        """
        full_path = os.path.abspath(model_path)
        if not os.path.isabs(model_path):
            # Try relative to models directory
            full_path = os.path.join(MODELS_DIR, model_path)

        if os.path.exists(full_path):
            logger.info(f"Loading weights from {full_path}")
            state_dict = torch.load(full_path, map_location="cpu")

            if "model" in state_dict:
                state_dict = state_dict["model"]

            # Handle different weight formats
            if any(k.startswith("dino.") for k in state_dict.keys()):
                logger.info("Loading full model weights (backbone + head)")
                self.load_state_dict(state_dict, strict=False)
            else:
                logger.info("Loading linear head weights only")
                self.head.load_state_dict(state_dict)
        else:
            logger.warning(f"Model weights not found at {full_path}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            images: Input images tensor (B, C, H, W) in range [0, 1]

        Returns:
            Classification logits (B, num_classes)
        """
        # Process images for DINOv3
        # Note: DINOv3 expects images in [0, 1] range, processor handles normalization
        dino_inputs = self.processor(images, return_tensors="pt", do_rescale=False)

        # Move inputs to same device as model
        device = next(self.parameters()).device
        dino_inputs = {k: v.to(device) for k, v in dino_inputs.items()}

        # Extract features from DINOv3
        with torch.no_grad():
            outputs = self.dino(**dino_inputs)

        # Use CLS token as features
        features = outputs.last_hidden_state[:, 0, :]  # (B, embedding_dim)

        # Classification
        logits = self.head(features)
        return logits

    def save(self, save_path: str) -> None:
        """
        Save model weights.

        Args:
            save_path: Path to save model weights
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model": self.state_dict(),
            "num_classes": self.num_classes,
            "dino_model_name": self.dino_model_name,
        }, save_path)
        logger.info(f"Model saved to {save_path}")

