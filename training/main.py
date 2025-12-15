import torch
from torchvision import transforms
import os, yaml
import mlflow
from dotenv import load_dotenv
from prometheus_client import start_http_server
from training.model import UNet
from training.trainer import Trainer
from training.dataset import SegmentationDataset
from training.utils import load_config_with_env
load_dotenv() # sets variables from .env

cfg_path = os.path.join(os.getcwd(), "training/train_config.yml")
config = load_config_with_env(cfg_path)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

mlflow.autolog(disable=True)

def main(config):
    device = config.get("device", "cpu")
    train_images = config.get("data", {}).get("train_images")
    train_labels = config.get("data", {}).get("train_labels")
    test_images = config.get("data", {}).get("test_images")
    test_labels = config.get("data", {}).get("test_labels")
    bs = config.get("batch_size", 8)
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 1e-3)
    step_size = config.get("lr_scheduler", {}).get("step_size", 3)
    gamma = config.get("lr_scheduler", {}).get("gamma", 0.5)
    img_size = config.get("img_size", [201,201])
    out_dir = config.get("out_dir", "./artifacts")
    exp_name = config.get("experiment_name", "default_experiment")

    # Dataset
    transform = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.GaussianBlur(3)
    ])
    train_ds = SegmentationDataset(images_dir=train_images, masks_dir=train_labels, img_size=img_size, transforms=transform)
    val_ds = SegmentationDataset(images_dir=test_images, masks_dir=test_labels, img_size=img_size)

    # Trainer
    model = UNet(n_channels=3, n_classes=1).to(device)
    trainer = Trainer(model=model, device=device, out_dir=out_dir, experiment_name=exp_name)
    try:
        with mlflow.start_run(run_name=exp_name):
            trainer.fit(
                train_dataset=train_ds,
                val_dataset=val_ds,
                epochs=epochs, 
                batch_size=bs,
                lr=lr, 
                step_size=step_size,
                gamma=gamma,
                save_best=True
            )
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    start_http_server(8000) # prom endpoint for scraping
    main(config)
