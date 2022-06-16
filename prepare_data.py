import argparse
import zipfile
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def parse_args() -> argparse:
    parser = argparse.ArgumentParser(
        "Split training dataset to train and test", add_help=False
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="number of test size"
    )
    parser.add_argument(
        "--img_dir", type=str, default="./training", help="directory of training folder"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="./training/label.csv",
        help="directory of label.csv folder",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="orchid_dataset",
        help="directory for train/test folder to save",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=22,
        help="controls the shuffling applied to the data before applying the split",
    )

    args = parser.parse_args()

    return args


def split_dataset(
    img_dir: str = "data",
    csv_dir: str = "label.csv",
    save_dir: str = "orchid_dataset",
    test_size: float = 0.2,
    random_state: int = None,
) -> None:

    # create subfolder for train/test
    if not Path.exists(Path(save_dir)):
        Path.mkdir(Path(save_dir))
        Path.mkdir(Path(save_dir) / "train")
        Path.mkdir(Path(save_dir) / "test")

    # 將資料分成 orchid_dataset/train 和 orchid_dataset/test
    df = pd.read_csv(csv_dir)
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["category"],
        shuffle=True,
        random_state=random_state,
    )
    train_save_path = Path(save_dir) / "train"
    test_save_path = Path(save_dir) / "test"

    for df_, save_dir_ in zip([train, test], [train_save_path, test_save_path]):
        label_list = df_["category"].unique()

        for label in tqdm(label_list):
            label_dir = Path(save_dir_) / str(label)
            if not Path.exists(label_dir):
                Path.mkdir(label_dir)

            sub_df = df_[df_["category"] == label]
            for f in sub_df["filename"]:
                img_path = str(Path(img_dir) / f)
                save_dir = str(Path(save_dir_) / str(label) / f)
                img = cv2.imread(img_path)

                cv2.imwrite(save_dir, img)

    train.to_csv(str(train_save_path) + ".csv", index=False)
    test.to_csv(str(test_save_path) + ".csv", index=False)


if __name__ == "__main__":

    # unzip training.zip if not exists training folder
    if not Path.exists(Path("training")):
        Path.mkdir(Path("training"))
        with zipfile.ZipFile("training.zip", "r") as f:
            f.extractall(".")

    args = parse_args()

    # create train/test folder
    split_dataset(
        img_dir=args.img_dir,
        csv_dir=args.csv_dir,
        save_dir=args.save_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
