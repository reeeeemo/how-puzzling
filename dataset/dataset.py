import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from glob import glob
import warnings
import cv2
import numpy as np

class PuzzleDataset(Dataset):
    def __init__(self, root_dir: str | Path, extension: str = "png", gray: bool=False, clahe: bool = False):
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
        
        self.gray = gray
        self.clahe = clahe

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        # get image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        
        # use contrast limited adaptive histogram equalization to improve contrast
        # divides img into smaller parts and adjusts contrast seperately
        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=5) 
            image = np.clip(clahe.apply(image) + 30, 0, 255).astype(np.uint8)

        image = Image.fromarray(image) # convert back to PIL for transforms

        # get label
        label_path = self.label_paths[idx]
        labels = []

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

        return image, labels
    
    def get_image_paths(self):
        return self.image_paths