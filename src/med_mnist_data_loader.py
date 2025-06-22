"""Medical MNIST Data Loader Module."""

import os
import sys
from typing import Tuple
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_parser import DataConfig


class SRFlatFolderDataset(Dataset):
    """Class for SR dataset with flat image folder structure.

    Loads images from a directory and generates paired low-resolution (LR)
    and high-resolution (HR) versions for supervised training. Supports
    configurable resizing and channel count.
    """
    def __init__(
        self, img_dir, lr_size=(28, 28),
        hr_size=(64, 64), img_channels=3
    ):
        """Initialize the super-resolution dataset from a flat image folder.

        Constructs low-resolution and high-resolution image pairs using
        torchvision transforms. Grayscale or RGB format is supported based
        on the specified number of image channels.

        Args:
            img_dir (str): Path to the folder containing input images.
            lr_size (tuple[int, int], optional): Target size for low-resolution
                images. Defaults to (28, 28).
            hr_size (tuple[int, int], optional): Target size for high-resolution
                images. Defaults to (64, 64).
            img_channels (int, optional): Number of output channels per image
                (1 for grayscale, 3 for RGB). Defaults to 3.
        """
        self.img_paths = [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_size),
            transforms.Grayscale(num_output_channels=img_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*img_channels, std=[0.5]*img_channels)
        ])

        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size),
            transforms.Grayscale(num_output_channels=img_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*img_channels, std=[0.5]*img_channels)
        ])

    def __len__(self) -> int:
        """Return the number of image samples in the dataset.

        Returns:
            int: Total number of image files found in the dataset directory.
        """
        return len(self.img_paths)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """Load and transform one image into LR and HR versions.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            tuple[Tensor, Tensor]: A pair of tensors (LR, HR) with shape
                (C, H, W), normalized to [-1, 1].
        """
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.lr_transform(img), self.hr_transform(img)

    pass


def get_med_mnist_train_val_loaders(
    img_dir, data_config: DataConfig
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for super-resolution datasets.

    Splits the dataset found at `img_dir/Train` into training and validation
    sets according to `train_percentage`.

    Args:
        img_dir (str): Path to dataset root folder.
        data_config (DataConfig): Configuration object containing batch size,
            image channels, and train/val split ratio.

    Returns:
        tuple[DataLoader, DataLoader]: DataLoaders for training and validation.
    """
    train_dir = os.path.join(img_dir, 'Train')

    dataset = SRFlatFolderDataset(
        img_dir=train_dir,
        lr_size=(28, 28),
        hr_size=(64, 64),
        img_channels=data_config.img_channels
    )

    train_size = int(
        len(dataset) * data_config.train_percentage
    )
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=data_config.batch_size,
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=data_config.batch_size,
        shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def get_med_mnist_test_loader(
    img_dir, data_config: DataConfig
) -> DataLoader:
    """Create test DataLoader for super-resolution evaluation.

    Loads images from `img_dir/Test` and prepares paired LR and HR image
    tensors for inference.

    Args:
        img_dir (str): Path to dataset root folder.
        data_config (DataConfig): Configuration with image channel count.

    Returns:
        DataLoader: A DataLoader for the test set with batch size 1.
    """
    test_dir = os.path.join(img_dir, 'Test')

    dataset = SRFlatFolderDataset(
        img_dir=test_dir,
        lr_size=(28, 28),
        hr_size=(64, 64),
        img_channels=data_config.img_channels
    )

    return DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=4
    )
