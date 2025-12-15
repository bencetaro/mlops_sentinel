import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml, os, shutil
from pathlib import Path
from training.dataset import SegmentationDataset
from inference.utils import make_dir
from inference.preparation import SentinelPreprocessor, SentinelPostprocessor
import mlflow

class UNetInference:
    def __init__(self, model_uri="models:/unet_model/Production", device='cpu'):
        self.device = torch.device(device)
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to(self.device)
        self.model.eval()

    def predict_dataset(self, dataset, output_folder, batch_size=8):
        output_folder = make_dir(output_folder)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                imgs = data[0] if isinstance(data, (list, tuple)) else data
                imgs = imgs.to(self.device)
                preds = torch.sigmoid(self.model(imgs))
                preds = (preds > 0.5).float()
                for i in range(len(preds)):
                    save_image(preds[i], output_folder / f'pred_{batch_idx}_{i}.png')

if __name__ == "__main__":
    cfg_path = Path("inference/inf_config.yml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    cwd = Path.cwd()
    temp_prep = make_dir(cwd / "temp_preprocessed")
    temp_preds = make_dir(cwd / "temp_predictions")

    device = config.get("device", "cpu")
    model_uri = config.get("model_uri", "models:/unet_model/Production")
    input_folder = Path(config.get("input_folder", "inference/input_tci"))
    output_folder = make_dir(config.get("output_folder", "inference/predictions"))
    batch_size = config.get("batch_size", 8)
    img_size = config.get("img_size", 201)
    crop_box = tuple(config.get("crop_box", (0,0,10854,10854)))

    # 1. Preprocess Sentinel images
    preprocessor = SentinelPreprocessor(
        input_folder=input_folder,
        output_folder=temp_prep,
        box=crop_box,
        tile_size=img_size,
    )
    preprocessor()

    # 2. Inference
    inference = UNetInference(model_uri, device)
    test_dataset = SegmentationDataset(
        images_dir=str(temp_prep),
        masks_dir=None,
        img_size=(img_size, img_size),
    )
    inference.predict_dataset(test_dataset, temp_preds, batch_size)

    # 3. Postprocess predictions
    # Use first TCI in folder for georeferencing
    tci_files = list(input_folder.glob("*"))
    if not tci_files:
        raise FileNotFoundError(f"No Sentinel images found in {input_folder}")
    postprocessor = SentinelPostprocessor(
        original_sentinel_path=tci_files[0],
        predictions_folder=temp_preds,
        output_folder=output_folder,
        apply_geoparsing=True,
    )
    postprocessor()

    # 4. Cleanup temp folders
    if temp_prep.exists():
        shutil.rmtree(temp_prep)
    if temp_preds.exists():
        shutil.rmtree(temp_preds)
    print("Temporary files cleaned up.")
