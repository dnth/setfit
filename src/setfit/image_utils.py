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


class ImageTransform:
    """Standard image transformation pipeline for TIMM models."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """Initialize image transformations.

        Args:
            image_size: Target size for images (height, width)
            is_training: Whether to apply data augmentation
            mean: Normalization mean values
            std: Normalization std values
        """
        self.image_size = image_size
        self.is_training = is_training
        self.mean = mean
        self.std = std

        # Base transforms (applied to all images)
        base_transforms = [
            transforms.Resize(image_size),
        ]

        if is_training:
            # Training augmentations
            train_transforms = [
                transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            ]
            base_transforms = train_transforms

        # Add normalization and tensor conversion
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.transform = transforms.Compose(base_transforms)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply transformations to an image.

        Args:
            image: PIL Image to transform

        Returns:
            Transformed image tensor
        """
        return self.transform(image)


def get_image_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    model_name: Optional[str] = None,
) -> ImageTransform:
    """Get appropriate image transforms for a TIMM model.

    Args:
        image_size: Target image size
        is_training: Whether for training or inference
        model_name: TIMM model name to get specific transforms

    Returns:
        Configured ImageTransform instance
    """
    # Model-specific mean/std values
    model_defaults = {
        "resnet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "efficientnet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "vit": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "deit": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "swin": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet defaults

    if model_name:
        for key in model_defaults:
            if key in model_name.lower():
                mean, std = model_defaults[key]
                break

    return ImageTransform(
        image_size=image_size,
        is_training=is_training,
        mean=mean,
        std=std,
    )


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
    transform: ImageTransform,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess a batch of images.

    Args:
        images: List of PIL Images
        transform: ImageTransform to apply
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
        num_classes: int = 0,  # 0 means use default output features
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize TIMM model wrapper.

        Args:
            model_name: Name of the TIMM model to use
            pretrained: Whether to use pretrained weights
            image_size: Input image size for the model
            num_classes: Number of output classes (0 for feature extraction)
            device: Device to load model on
        """
        self.model_name = model_name
        self.image_size = image_size
        self.num_classes = num_classes

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        if num_classes > 0:
            # Classification model
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        else:
            # Feature extraction model
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # This removes the classification head
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size[0], image_size[1], device=self.device)
            self.feature_dim = self.model(dummy_input).shape[1]

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Batch of images, shape (batch_size, 3, H, W)

        Returns:
            Features tensor of shape (batch_size, feature_dim)
        """
        images = images.to(self.device)

        with torch.no_grad():
            features = self.model(images)

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


def list_available_models(filter: Optional[str] = None) -> List[str]:
    """List available TIMM models, optionally filtered.

    Args:
        filter: String to filter model names

    Returns:
        List of model names
    """
    models = timm.list_models(pretrained=True)

    if filter:
        models = [m for m in models if filter.lower() in m.lower()]

    return sorted(models)


def recommend_models(task: str = "classification", min_params: int = 1000000) -> List[str]:
    """Recommend TIMM models for a specific task.

    Args:
        task: Task type ("classification", "detection", etc.)
        min_params: Minimum number of parameters

    Returns:
        List of recommended model names
    """
    if task == "classification":
        # Popular models for image classification
        candidates = [
            "resnet50",
            "resnet101",
            "efficientnet_b0",
            "efficientnet_b3",
            "vit_base_patch16_224",
            "deit_base_patch16_224",
            "swin_base_patch4_window7_224",
            "convnext_base",
        ]
    else:
        candidates = timm.list_models(pretrained=True)[:20]  # Top 20 models

    # Filter by parameter count if specified
    if min_params > 0:
        filtered = []
        for model_name in candidates:
            try:
                info = get_model_info(model_name)
                if info.get("param_count", 0) >= min_params:
                    filtered.append(model_name)
            except:
                continue
        return filtered

    return candidates