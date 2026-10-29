import torch
from torch.utils.data import Dataset
import numpy as np

PATCH_SIZE = 512

# ImageNet normalization for the pretrained ResNet encoder.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def patch_starts(length, patch_size):
    """Cover an entire dimension, including its far edge."""
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, patch_size))
    last = length - patch_size

    if starts[-1] != last:
        starts.append(last)

    return starts


class BuildingDataset(Dataset):
    def __init__(self, files):
        self.files = list(files)
        self.samples = []

        # Precompute patch coordinates without loading all images.
        for path in self.files:
            with np.load(path) as data:
                height, width = data["mask"].shape

            ys = patch_starts(height, PATCH_SIZE)
            xs = patch_starts(width, PATCH_SIZE)

            for y in ys:
                for x in xs:
                    self.samples.append((path, y, x))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, y, x = self.samples[index]

        with np.load(path) as data:
            image = data["image"][y:y + PATCH_SIZE,
                                  x:x + PATCH_SIZE].copy()
            mask = data["mask"][y:y + PATCH_SIZE,
                                x:x + PATCH_SIZE].copy()

        # HWC -> CHW. No contrast enhancement or augmentation.
        image = image.astype(np.float32)

        # Convert integer-style pixel values to [0, 1].
        # The source TIFFs are expected to contain 8-bit RGB values.
        image /= 255.0

        image = (image - MEAN) / STD
        image = np.moveaxis(image, -1, 0).copy()

        # U-Net output: one logit per pixel.
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return image, mask
