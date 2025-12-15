import os, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from prometheus_client import Gauge, Counter
from training.model import UNet
from training.utils import log_prediction_sample, dice

class Trainer:
    def __init__(self, model: UNet, device:str='cpu', out_dir:str='./outputs', experiment_name:str='default_experiment'):
        self.model = model
        self.device = torch.device(device)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.prom_epoch = Counter('epoch', 'Current Epoch', ['experiment'])
        self.prom_train_loss = Gauge('train_loss', 'Training Loss', ['experiment'])
        self.prom_val_loss = Gauge('val_loss', 'Validation Loss', ['experiment'])
        self.prom_val_dice = Gauge('val_dice', 'Validation Dice Score', ['experiment'])

        if self.device.type == 'cpu':
            torch.set_num_threads(8)
            torch.backends.cudnn.enabled = False

    def fit(self, train_dataset, val_dataset, epochs=8, batch_size=8, lr=1e-3, step_size=3, gamma=0.5, save_best=True):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "step_size": step_size,
            "gamma": gamma,
            "device": str(self.device),
            "torch_version": torch.__version__,
        })
        self.model.to(self.device)
        best_val = np.inf

        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for imgs, masks in train_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                preds = self.model(imgs)
                loss = sigmoid_focal_loss(preds, masks, alpha=0.25, gamma=2)
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
                train_losses.append(loss.mean().item())

            self.model.eval()
            val_losses = []
            dice_scores = []
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)
                    preds = self.model(imgs)
                    loss = sigmoid_focal_loss(preds, masks, alpha=0.25, gamma=2)                        
                    val_losses.append(loss.mean().item())
                    dice_scores.append(dice(preds, masks))

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_losses))
            avg_dice = float(np.mean(dice_scores))
            print(f"Epoch {epoch+1}/{epochs} - train {avg_train:.4f} - val {avg_val:.4f} - dice {avg_dice:.2f}")

            self.prom_epoch.labels(experiment=self.experiment_name).inc()
            self.prom_train_loss.labels(experiment=self.experiment_name).set(avg_train)
            self.prom_val_loss.labels(experiment=self.experiment_name).set(avg_val)
            self.prom_val_dice.labels(experiment=self.experiment_name).set(avg_dice)

            mlflow.log_metrics({
                "train_loss": avg_train,
                "val_loss": avg_val,
                "val_dice": avg_dice,
            }, step=epoch)

            pred_path = log_prediction_sample(imgs[0], masks[0], preds[0], epoch)
            mlflow.log_artifact(pred_path, artifact_path="prediction_samples")
            print(f"Logged prediction sample to: {pred_path}")

            if save_best and avg_val <= best_val:
                best_val = avg_val
                save_path = os.path.join(self.out_dir, "best_model.pt")
                torch.save(self.model.state_dict(), str(save_path))
                print(f"Saved a better model (val loss: {avg_val:.4f}): {save_path}")
                self._register_and_stage_model(model_name="unet_model")
                print(f"Model logged to MLflow under experiment: {self.experiment_name}")

            scheduler.step()

    def _register_and_stage_model(self, model_name="unet"):
        # Log model to local MLflow
        cpu_model = self.model.to('cpu')
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id

        mlflow.pytorch.log_model(
            pytorch_model=cpu_model,
            name=model_name,
            pip_requirements=["torch==2.5.1+cpu","torchvision==0.20.1+cpu"]
        )
        print(f"Model logged to Mlflow locally under name: {model_name}")

        model_uri = f"runs:/{run_id}/{model_name}"
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Registered model '{model_name}' version {registered_model.version} -> Production")
