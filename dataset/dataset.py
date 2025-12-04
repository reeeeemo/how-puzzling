import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from glob import glob
import warnings

class PuzzleDataset(Dataset):
    def __init__(self, root_dir: str | Path, extension: str = "png", transform=None):
        # get root dir and image / label paths
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            warnings.warn(f"Root directory ({root_dir}) does not exist.")

        self.image_paths = sorted(glob(str(self.root_dir / "images" / "train" / "**" / f"*.{extension}"), recursive=True))
        self.label_paths = sorted(glob(str(self.root_dir / "labels" / "train" / "**" / "*.txt"), recursive=True))
        len_image, len_lbl = len(self.image_paths), len(self.label_paths)
        if len_image != len_lbl:
            warnings.warn(f"Image ({len_image}) and Label ({len_lbl}) directories have unequal elements.")
        if not self.image_paths or not self.label_paths:
            warnings.warn(f"Image path exists: ({bool(self.image_paths)}). Label path exists ({bool(self.label_paths)})")

        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        # get image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L") #.convert("RGB") # 
        image = image.resize((1920, 1080), Image.LANCZOS)

        # get label
        label_path = self.label_paths[idx]
        labels = []

        # TODO: convert cls, cx, cy, w, h into pixel coords then apply transforms and convert back to normalized format
        # assuming yolov8-v11 bounding box format
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line: # empty
                        continue
                    values = line.split(" ")
                    if len(values) != 5:
                        warnings.warn(f"Invalid label format in {label_path}: expected 5 vals, got {len(values)}")
                        continue
                    labels.append(torch.tensor([float(v) for v in values]))
        except Exception as e:
            warnings.warn(f"Unexpected error occured for file {label_path}: {e}")
            return image, []

        if self.transform:
            image = self.transform(image)

        return image, labels