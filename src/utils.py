import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str):

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, total_acc = 0, 0
    outputs, labels = [], []

    for _, (data, label) in enumerate(dataloader):
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)
        y_true = label.cpu().numpy()
        y_pred = torch.argmax(output, 1).cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        total_loss += loss
        total_acc += acc
        outputs.append(y_pred)
        labels.append(y_true)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)

    outputs = np.concatenate(outputs)
    labels = np.concatenate(labels)
    macro_f1 = f1_score(outputs, labels, average="macro")

    final_score = 0.5 * (avg_acc + macro_f1)

    return avg_loss, avg_acc, macro_f1, final_score


@torch.no_grad()
def predict(model, dataloader, device, class_mapping) -> list[int]:

    label_predictions_list = []

    for _, img in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = img.to(device)
        output = model(img)

        y_pred = torch.argmax(output, 1).cpu().numpy()

        # label encode
        for y_ in y_pred:
            y = class_mapping[y_]
            label_predictions_list.append(y)

    return label_predictions_list
