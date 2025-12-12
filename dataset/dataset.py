import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from glob import glob
import warnings
import cv2
import numpy as np

class PuzzleDataset(Dataset):
    """
        Dataset of Jigsaw Puzzle pieces given a YOLO-style folder schema of
        images and optional bounding box labels.
    """
    def __init__(self, root_dir: str | Path, extension: str = "png", gray: bool=False, clahe: bool = False):
        # get root dir and image / label paths of train and val splits
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            warnings.warn(f"Root directory ({root_dir}) does not exist.")

        self.images = {
            "train": sorted(glob(str(self.root_dir / "images" / "train" / "**" / f"*.{extension}"), recursive=True)),
            "val": sorted(glob(str(self.root_dir / "images" / "val" / "**" / f"*.{extension}"), recursive=True))
        }
        self.labels = {
            "train": sorted(glob(str(self.root_dir / "labels" / "train" / "**" / "*.txt"), recursive=True)),
            "val": sorted(glob(str(self.root_dir / "labels" / "val" / "**" / "*.txt"), recursive=True))
        }
        for split in ["train", "val"]:
            img_path = self.images[split]
            lbl_path = self.labels[split]
            len_image, len_lbl = len(img_path), len(lbl_path)
            if len_image != len_lbl:
                warnings.warn(f"{split} Image ({len_image}) and Label ({len_lbl}) directories have unequal elements.")
            if not img_path or not lbl_path:
                warnings.warn(f"{split} Image path exists: ({bool(img_path)}). Label path exists ({bool(lbl_path)})")
        
        self.gray = gray
        self.clahe = clahe
        self.current_split = "train"

    def __len__(self):
        return len(self.images[self.current_split])
    
    def set_split(self, split: str = "train"):
        if split not in ["train", "val"]:
            warnings.warn("Illegal split used.")
            return
        self.current_split = split
    
    def __getitem__(self, idx: int):
        # get train image
        img_path = self.images[self.current_split][idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        
        # use contrast limited adaptive histogram equalization to improve contrast
        # divides img into smaller parts and adjusts contrast seperately
        if self.clahe and self.gray: # clahe expects single channel
            clahe = cv2.createCLAHE(clipLimit=5) 
            image = np.clip(clahe.apply(image) + 30, 0, 255).astype(np.uint8)
        elif self.clahe and not self.gray:
            warnings.warn("CLAHE requires grayscale (single channel input)")
        
        image = Image.fromarray(image) # convert back to PIL for transforms

        if (idx >= len(self.labels[self.current_split])):
            return image, []
        # get label
        label_path = self.labels[self.current_split][idx]
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
        return self.images[self.current_split]