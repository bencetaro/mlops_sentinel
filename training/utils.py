import os, yaml
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path

def make_transforms(img_size=(201,201), training: bool = True):
    base = [transforms.ToPILImage()]
    if training:
        base += [
            transforms.ColorJitter(0.1,0.1,0.1,0.1),
            transforms.GaussianBlur(3, sigma=(0.1,2.0)),
        ]
    base += [transforms.Resize(img_size), transforms.ToTensor()]
    return transforms.Compose(base)

def load_config_with_env(path: str):
    with open(path, "r") as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    config = yaml.safe_load(expanded)
    return config

def log_prediction_sample(image, mask, pred_mask, epoch, out_dir="training/outputs/predictions"):
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    mask = mask.detach().cpu().squeeze().numpy()
    if mask.max() > 1:
        mask = mask / 255.0
    pred_prob = torch.sigmoid(pred_mask).detach().cpu().squeeze().numpy()
    pred_bin = (pred_prob > 0.5).astype(np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")
    ax[2].imshow(pred_prob, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("Prediction (Prob)")
    ax[2].axis("off")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(out_dir) / f"sample_epoch_{epoch}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

def iou(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + smooth) / (union + smooth))

def dice(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return float((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
