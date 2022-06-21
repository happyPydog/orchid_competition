import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(
    original_image, attribution_generator, image_size, class_index=None
):
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).cuda(),
        method="transformer_attribution",
        index=class_index,
    ).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 24, 24)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode="bilinear"
    )
    transformer_attribution = (
        transformer_attribution.reshape(image_size, image_size)
        .cuda()
        .data.cpu()
        .numpy()
    )
    transformer_attribution = (
        transformer_attribution - transformer_attribution.min()
    ) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (
        image_transformer_attribution - image_transformer_attribution.min()
    ) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def print_top_classes(predictions, class_mapping):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(class_mapping[cls_idx])
        if len(class_mapping[cls_idx]) > max_str_len:
            max_str_len = len(class_mapping[cls_idx])

    print("Top 5 classes:")
    top_5_class_prob = []
    for cls_idx in class_indices:
        output_string = f"{class_mapping[cls_idx]} : {100 * prob[0, cls_idx]:.1f}%"
        print(output_string, cls_idx)
        top_5_class_prob.append(output_string)

    return ", ".join(top_5_class_prob)


def plot_attention(
    model,
    attribution_generator,
    image_dir: str,
    image_size: int,
    transform: transforms,
    device: int,
    class_mapping: dict,
    class_index: int = None,
    save_dir: str = None,
):

    # laod image from image_dir
    image = Image.open(image_dir)
    transform_image = transform(image)
    fig, axes = plt.subplots(1, 2, figsize=(6, 8))

    # original image
    axes[0].imshow(image)
    axes[0].axis("off")

    output = model(transform_image.unsqueeze(0).to(device))
    class_prob = print_top_classes(output, class_mapping)
    fig.suptitle(class_prob, fontsize=16)

    # predicted class
    attn_mapping = generate_visualization(
        transform_image, attribution_generator, image_size, class_index=class_index
    )
    axes[1].imshow(attn_mapping)
    axes[1].axis("off")

    if save_dir:
        plt.savefig(f"{save_dir}.png")
