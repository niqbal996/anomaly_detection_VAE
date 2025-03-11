import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class PhenoBenchDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int],
        split: str = "train",
        transform = None
    ):
        """
        Args:
            root_dir (str): Root directory of the PhenoBench dataset
            img_size (tuple): Desired output image size (height, width)
            split (str): Dataset split ('train', 'val', 'test')
            transform: Optional transform to be applied to images
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split
        self.transform = transform

        # Get image paths (both png and jpg)
        self.image_dir = self.root_dir / split / "images"
        self.image_paths = []
        self.image_paths.extend(glob.glob(str(self.image_dir / "*.png")))
        self.image_paths.extend(glob.glob(str(self.image_dir / "*.jpg")))
        self.image_paths.sort()  # Ensure consistent ordering

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx (int): Index of the sample to fetch

        Returns:
            torch.Tensor: The image tensor
        """
        img_path = self.image_paths[idx]
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image
