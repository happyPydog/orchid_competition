import json
import argparse
import warnings
import pandas as pd
from torch.utils.data import DataLoader
import torch
import timm

from src.transform import test_tf
from src.dataset import OrchidDataset
from src.utils import predict

warnings.filterwarnings("ignore")


def parse_args() -> argparse:
    parser = argparse.ArgumentParser("Inference", add_help=False)

    # data configuration
    parser.add_argument("--IMAGE_DIR", type=str, default="test_dataset")
    parser.add_argument("--SUBMISSION_DIR", type=str, default="submission_template.csv")
    parser.add_argument("--SUBMISSION_SAVE_DIR", type=str, default="submission.csv")
    parser.add_argument("--CLASS_MAPPING", type=str, default="class_mapping.json")
    parser.add_argument("--IMAGE_SIZE", type=int, default=384)

    # Dataloader configuration
    parser.add_argument("--BATCH_SIZE", type=int, default=32)
    parser.add_argument("--NUM_WORKWERS", type=int, default=4)

    # model configuration
    parser.add_argument(
        "--CHECKPOINT",
        type=str,
        default="swinv2_base_window12to24_192to384_22kft1k.pt",
    )
    parser.add_argument(
        "--MODEL_NAME",
        type=str,
        default="swinv2_base_window12to24_192to384_22kft1k",
    )

    args = parser.parse_args()

    return args


def main(args):

    # load class to index json file
    with open(args.CLASS_MAPPING, "r") as f:
        class_mapping = json.load(f)
    class_mapping = {int(k): v for k, v in class_mapping.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.SUBMISSION_DIR)
    transform = test_tf(args.IMAGE_SIZE)
    dataset = OrchidDataset(df=df, img_dir=args.IMAGE_DIR, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=args.NUM_WORKWERS,
        shuffle=False,
    )

    checkpoint = torch.load(args.CHECKPOINT)
    model = timm.create_model(
        args.MODEL_NAME,
        pretrained=False,
        num_classes=len(class_mapping),
        img_size=args.IMAGE_SIZE,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # prediction of public and private dataset
    predictions = predict(model, dataloader, device, class_mapping)

    df["category"] = predictions
    df.to_csv(args.SUBMISSION_SAVE_DIR, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
