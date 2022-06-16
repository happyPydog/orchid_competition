import time
import argparse
import warnings
from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils.clip_grad import dispatch_clip_grad

from src.transform import train_tf, test_tf
from src.utils import evaluate

warnings.filterwarnings("ignore")


def parse_args() -> argparse:
    parser = argparse.ArgumentParser("Training model", add_help=False)

    # data settings
    parser.add_argument(
        "--TRAIN_DIR", type=str, default="orchid_dataset/train", help="Train directory"
    )
    parser.add_argument(
        "--TEST_DIR", type=str, default="orchid_dataset/test", help="Test directory"
    )
    parser.add_argument("--IMAGE_SIZE", type=int, default=384, help="Input image size")
    parser.add_argument("--COLOR_JITTER", type=float, default=0.3)

    # model settings
    parser.add_argument(
        "--MODEL_NAME",
        type=str,
        default="swinv2_base_window12to24_192to384_22kft1k",
        help="Model name",
    )
    parser.add_argument("--FINE_TUNE", type=bool, default=True)
    parser.add_argument(
        "--CHECKPOINT", type=str, default="swinv2_base_window12_192_22k.pt"
    )

    # model parameters
    parser.add_argument("--DROP_RATE", type=float, default=0.1)
    parser.add_argument("--ATTN_DROP_RATE", type=float, default=0.1)
    parser.add_argument("--DROP_PATH_RATE", type=float, default=0.1)
    parser.add_argument("--NUM_CLASSES", type=int, default=219)

    # optimizer parameters
    parser.add_argument("--LEARNING_RATE", type=float, default=3e-3)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.05)

    # scheduler
    parser.add_argument("--LR_MIN", type=float, default=1e-5)
    parser.add_argument("--T_INITIAL", type=int, default=10)
    parser.add_argument("--WARMUP_T", type=int, default=5)
    parser.add_argument("--WARMUP_LR_INIT", type=float, default=1e-5)
    parser.add_argument("--K_DECAY", type=float, default=0.75)

    # training settings
    parser.add_argument("--EPOCHS", type=int, default=200)
    parser.add_argument("--BATCH_SIZE", type=int, default=2)
    parser.add_argument("--CLIP_GRAD", type=float, default=5.0)
    parser.add_argument("--ONLY_TRAINING_MSA", type=bool, default=True)
    parser.add_argument("--LABEL_SMOOTHING", type=float, default=0.1)

    args = parser.parse_args()

    return args


# training configuration


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = {
        "train": train_tf(
            args.IMAGE_SIZE, three_data_aug=True, color_jitter=args.COLOR_JITTER
        ),
        "val": test_tf(args.IMAGE_SIZE),
    }

    train_dataset = ImageFolder(root=args.TRAIN_DIR, transform=transform["train"])
    val_dataset = ImageFolder(root=args.TEST_DIR, transform=transform["val"])
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=True)

    model = timm.create_model(
        args.MODEL_NAME,
        pretrained=False,
        img_size=args.IMAGE_SIZE,
        num_classes=args.NUM_CLASSES,
        drop_rate=args.DROP_RATE,
        attn_drop_rate=args.ATTN_DROP_RATE,
        drop_path_rate=args.DROP_PATH_RATE,
    )

    if args.FINE_TUNE:
        state_dict = model.state_dict()
        checkpoint = torch.load(args.CHECKPOINT, map_location="cpu")
        checkpoint_model = checkpoint["model"]

        pre_trained_layers = {}
        for k, v in checkpoint_model.items():
            if checkpoint_model[k].shape == state_dict[k].shape:
                pre_trained_layers[k] = v

        model.load_state_dict(pre_trained_layers, strict=False)

    if args.ONLY_TRAINING_MSA:

        # only train MSA(multi-head self-attention) and head
        for name_p, p in model.named_parameters():
            if ".attn." in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False

        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True

        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print("no patch embed")

    print(
        f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    model.to(device)

    criterion = LabelSmoothingCrossEntropy(args.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY
    )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.T_INITIAL,
        warmup_t=args.WARMUP_T,
        warmup_lr_init=args.WARMUP_LR_INIT,
        k_decay=args.K_DECAY,
        lr_min=args.LR_MIN,
    )

    for epoch in range(args.EPOCHS):
        start_time = time.time()
        for i, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            # forward pass
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            dispatch_clip_grad(model.parameters(), args.CLIP_GRAD)

            # gradient decent or adam step
            optimizer.step()

        # update scheduler
        scheduler.step_update(epoch)

        # if (epoch + 1) % 2 == 0:

        # Print current metric
        train_loss, train_acc, train_macro_f1, train_final_scroe = evaluate(
            model, train_loader, device
        )
        val_loss, val_acc, val_macro_f1, val_final_scroe = evaluate(
            model, val_loader, device
        )

        print(
            f"【Epoch={epoch+1}】 train:【loss={train_loss:.3f}, acc={100*train_acc:.2f}%, f1={train_macro_f1:.3f}, final={train_final_scroe:.3f}】 \
            val: 【loss={val_loss:.3f}, acc={100*val_acc:.2f}%, f1={val_macro_f1:.3f}, final={val_final_scroe:.3f}】 {(time.time() - start_time):.2f}/s"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            f=f"{args.MODEL_NAME}.pt",
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
