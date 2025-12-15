import rasterio
from PIL import Image
import numpy as np
from glob import glob
from typing import Tuple
from pathlib import Path
from itertools import product
from inference.utils import sorted_alphanumeric, make_dir

Image.MAX_IMAGE_PIXELS = None 

class SentinelPreprocessor:
    def __init__(self, input_folder: str, output_folder: str, box: Tuple[int,int,int,int]=(0,0,10854,10854), tile_size=201):
        self.input_folder = Path(input_folder)
        self.output_folder = make_dir(output_folder)
        self.box = box
        self.tile_size = tile_size

    def __call__(self):
        for sentinel_file in self.input_folder.glob("*"):
            self.crop_to_extent(sentinel_file)
            self.split_to_tiles(sentinel_file)
        print("Preprocessing complete.")

    def crop_to_extent(self, input_file: Path):
        im = Image.open(input_file)
        cropped = im.crop(self.box)
        out_path = self.output_folder / input_file.name
        cropped.save(out_path, dpi=(300, 300))
        return out_path

    def split_to_tiles(self, input_file: Path):
        cropped_file = self.output_folder / input_file.name
        img = Image.open(cropped_file)
        w, h = img.size
        for i, j in product(range(0, h - h % self.tile_size, self.tile_size),
                            range(0, w - w % self.tile_size, self.tile_size)):
            tile = img.crop((j, i, j + self.tile_size, i + self.tile_size))
            tile_name = f"{input_file.stem}_{i}_{j}.jpg"
            tile.save(self.output_folder / tile_name, dpi=(300, 300))

class SentinelPostprocessor:
    def __init__(self, original_sentinel_path: str, predictions_folder: str, output_folder: str, apply_geoparsing: bool = True):
        self.original_sentinel_path = Path(original_sentinel_path)
        self.predictions_folder = Path(predictions_folder)
        self.output_folder = make_dir(output_folder)
        self.apply_geoparsing = apply_geoparsing
        self.prediction_files = sorted_alphanumeric(glob(str(self.predictions_folder / "*")))
        self.concatenated_prediction = self.output_folder / "concatenated_prediction.jpg"
        self.georeferenced_prediction = self.output_folder / "geoparsed_prediction.tif"

    def __call__(self):
        self.concat_tiles()
        print("Tile concatenation done.")
        if self.apply_geoparsing:
            self.parse_georef()
            print("Geoparsing done.")

    def concat_tiles(self):
        if not self.prediction_files:
            raise ValueError("No prediction tiles found.")

        images = [Image.open(p) for p in self.prediction_files]
        # Determine grid size
        widths, heights = zip(*(i.size for i in images))
        tile_w, tile_h = widths[0], heights[0]
        # Assume tiles are square grid
        tiles_per_row = int(np.sqrt(len(images)))
        tiles_per_col = len(images) // tiles_per_row

        new_im = Image.new('RGB', (tile_w * tiles_per_row, tile_h * tiles_per_col))
        for idx, im in enumerate(images):
            row = idx // tiles_per_row
            col = idx % tiles_per_row
            new_im.paste(im, (col * tile_w, row * tile_h))
        new_im.save(self.concatenated_prediction, dpi=(300, 300))

    def parse_georef(self, dtype='float32'):
        with rasterio.open(self.concatenated_prediction) as pred_ds:
            p_image = pred_ds.read() # shape: (bands, height, width)

        with rasterio.open(self.original_sentinel_path) as tci_ds:
            transform = tci_ds.transform
            crs = tci_ds.crs

        with rasterio.open(
            self.georeferenced_prediction,
            'w',
            driver='GTiff',
            height=p_image.shape[1],
            width=p_image.shape[2],
            count=p_image.shape[0],
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(p_image.astype(dtype))
        print(f"Georeferenced prediction saved at: {self.georeferenced_prediction}")
