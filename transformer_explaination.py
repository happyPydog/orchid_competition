""" Transformer Interpretability Beyond Attention Visualization
"""

import json
import torch
from transformer_explainability.ViT_LRP import (
    vit_base_patch16_224,
    vit_base_patch16_384,
)
from transformer_explainability.ViT_explaination_generator import LRP
from transformer_explainability.utils import plot_attention
from src.transform import test_tf

# configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "image/1lebnyzs98.jpg"
CHECKPOINT_DIR = "vit_base_patch16_384.pt"
IMAGE_SIZE = 384
TRANSFORM = test_tf(IMAGE_SIZE)


def main():

    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    class_mapping = {int(k): v for k, v in class_mapping.items()}

    model = vit_base_patch16_384(pretrained=False, img_size=IMAGE_SIZE, num_classes=219)

    checkpoint = torch.load(CHECKPOINT_DIR)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    attribution_generator = LRP(model)

    plot_attention(
        model,
        attribution_generator,
        image_dir=IMAGE_DIR,
        image_size=IMAGE_SIZE,
        transform=TRANSFORM,
        device=DEVICE,
        class_mapping=class_mapping,
        class_index=None,
        save_dir="image/1lebnyzs98_explaination",
    )


if __name__ == "__main__":
    main()
