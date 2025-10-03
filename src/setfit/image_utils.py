"""
Image utilities for SetFit TIMM integration.

This module provides utilities for loading, preprocessing, and handling images
for use with TIMM models in the SetFit framework.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
from timm.data import resolve_model_data_config, create_transform


def get_image_transforms(
    model_name: str,
    is_training: bool = True,
) -> transforms.Compose:
    """Get appropriate image transforms for a TIMM model.

    Args:
        model_name: TIMM model name to get specific transforms
        is_training: Whether for training or inference

    Returns:
        Configured transform
    """
    dummy_model = timm.create_model(model_name, pretrained=False, num_classes=0)
    data_config = resolve_model_data_config(dummy_model)
    return create_transform(**data_config, is_training=is_training)


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an image from disk.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be opened
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Could not open image {image_path}: {e}")


def load_images_from_directory(
    directory: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> List[Tuple[Path, Image.Image]]:
    """Load all images from a directory.

    Args:
        directory: Directory containing images
        extensions: Valid image file extensions

    Returns:
        List of (path, image) tuples
    """
    directory = Path(directory)
    images = []

    for ext in extensions:
        for image_path in directory.rglob(f"*{ext}"):
            try:
                image = load_image(image_path)
                images.append((image_path, image))
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load {image_path}: {e}")
                continue

    return images


def create_image_dataset_from_directory(
    dataset_dir: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> Dict[str, List[Path]]:
    """Create a dataset structure from a directory of images.

    Assumes directory structure like:
    dataset_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image3.jpg
            image4.jpg

    Args:
        dataset_dir: Root directory of the dataset
        extensions: Valid image file extensions

    Returns:
        Dictionary mapping class names to lists of image paths
    """
    dataset_dir = Path(dataset_dir)
    dataset = {}

    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        image_paths = []

        for ext in extensions:
            image_paths.extend(class_dir.glob(f"*{ext}"))

        if image_paths:
            dataset[class_name] = sorted(image_paths)

    return dataset


def preprocess_image_batch(
    images: List[Image.Image],
    transform: transforms.Compose,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess a batch of images.

    Args:
        images: List of PIL Images
        transform: Transform to apply
        device: Device to move tensors to

    Returns:
        Batch tensor of shape (batch_size, channels, height, width)
    """
    if not images:
        raise ValueError("No images provided")

    # Apply transforms to each image
    transformed_images = [transform(img) for img in images]

    # Stack into batch
    batch = torch.stack(transformed_images, dim=0)

    if device is not None:
        batch = batch.to(device)

    return batch


class TimmModelWrapper:
    """Wrapper for TIMM models to provide consistent interface."""

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        num_classes: int = 0,  # Ignored for feature extraction
        device: Optional[Union[str, torch.device]] = None,
        train_embeddings: bool = False,
    ):
        """Initialize TIMM model wrapper.

        Args:
            model_name: Name of the TIMM model to use
            pretrained: Whether to use pretrained weights
            image_size: Input image size for the model
            num_classes: Ignored for feature extraction
            device: Device to load model on
            train_embeddings: Whether to train the model embeddings (if False, model will be frozen)
        """
        self.model_name = model_name
        self.image_size = image_size
        self.num_classes = num_classes
        self.train_embeddings = train_embeddings

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model for feature extraction
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
        )

        self.model = self.model.to(self.device)

        # Freeze model if not training embeddings
        if not train_embeddings:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size[0], image_size[1], device=self.device)
            dummy_features = self.model.forward_features(dummy_input)
            dummy_output = self.model.forward_head(dummy_features, pre_logits=True)
            self.feature_dim = dummy_output.shape[1]

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Batch of images, shape (batch_size, 3, H, W)

        Returns:
            Features tensor of shape (batch_size, feature_dim)
        """
        images = images.to(self.device)

        # Use gradients only during training when train_embeddings=True
        # Use no_grad during inference to save memory
        if self.train_embeddings and self.model.training:
            # Training mode with trainable embeddings - allow gradients
            features = self.model.forward_features(images)
            features = self.model.forward_head(features, pre_logits=True)
        else:
            # Inference mode or frozen embeddings - use no_grad for memory efficiency
            with torch.no_grad():
                features = self.model.forward_features(images)
                features = self.model.forward_head(features, pre_logits=True)

        return features

    def get_image_size(self) -> Tuple[int, int]:
        """Get the expected image size for this model."""
        return self.image_size

    def get_feature_dim(self) -> int:
        """Get the feature dimension."""
        return self.feature_dim

    def to(self, device: Union[str, torch.device]) -> "TimmModelWrapper":
        """Move model to device."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def train(self):
        """Set model to training mode."""
        if self.train_embeddings:
            self.model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()


def get_model_info(model_name: str) -> Dict[str, any]:
    """Get information about a TIMM model.

    Args:
        model_name: Name of the TIMM model

    Returns:
        Dictionary with model information
    """
    try:
        # Try to create a small instance to get info
        model = timm.create_model(model_name, pretrained=False, num_classes=0)

        info = {
            "model_name": model_name,
            "param_count": sum(p.numel() for p in model.parameters()),
            "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        # Try to get input size
        try:
            info["input_size"] = model.default_cfg.get("input_size", (3, 224, 224))
        except:
            info["input_size"] = (3, 224, 224)

        return info

    except Exception as e:
        return {"model_name": model_name, "error": str(e)}
