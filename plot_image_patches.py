from msilib.schema import Patch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

os.environ["TF_CPPMIN_LOG_LEVEL"] = "2"


def read_image(image_file: str, transform: transforms) -> torch.Tensor:
    """Read image from image directory"""
    image = Image.open(image_file)

    if transform:
        image = transform(image)

    return image


def create_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Create patches from image"""
    C, H, W = image.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), "Image H, W must be divide by patch size."
    image = image.unsqueeze(0)

    unfold = nn.Unfold(
        kernel_size=patch_size,
        stride=patch_size,
        dilation=1,
        padding=0,
    )
    patches = unfold(image)  # (1, 49152, 9)
    patches = patches.permute(0, 2, 1)  # (1, 9, 49152)

    return patches


def render_patches(patches: torch.Tensor, patch_size: int) -> None:
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(8, 8))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        image = patch.reshape(-1, patch_size, patch_size)
        image = transforms.ToPILImage()(image)
        ax.imshow(image)
        ax.axis("off")

    plt.savefig("image/patches_1lebnyzs98.png")


def render_flat(patches: torch.Tensor, patch_size: int) -> None:
    plt.figure(figsize=(12, 2))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(1, 10, i + 1)
        image = patch.reshape(-1, patch_size, patch_size)
        image = transforms.ToPILImage()(image)
        ax.imshow(image)
        ax.axis("off")

    plt.savefig("image/flatten_patches_1lebnyzs98.png")


def main():

    iamge_dir = "image/1lebnyzs98.jpg"
    image_size = 384
    patch_size = 128
    image = read_image(
        iamge_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        ),
    )  # (3, 384, 384)

    patches = create_patches(image, patch_size)  # (1, 9, 49152)
    render_patches(patches, patch_size)
    render_flat(patches, patch_size)


if __name__ == "__main__":
    main()
