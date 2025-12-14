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

        Intended for inference and dataset generation, 
        supports optional grayscale / CLAHE preprocessing.
        
        Attributes:
            splits (list[str]): list of legal splits for dataset
            current_split (str): active dataset split
            root_dir (Path): root dataset directory
            images (dict[str, list[str]]): image paths per split
            labels (dict[str, list[str]]): labels paths per split
            gray (bool): whether images are converted to grayscale
            clahe (bool): whether CLAHE preprocessing is applied
    """
    def __init__(self, 
            splits: list[str],
            root_dir: str | Path, 
            extension: str = "png", 
            gray: bool = False, 
            clahe: bool = False
        ):
        # get root dir and image / label paths of train and val splits
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            warnings.warn(f"Root directory ({root_dir}) does not exist.")
        if len(splits) <= 0:
            warnings.warn("Invalid splits given. Please instantiate with at least 1 split")
            return
        
        # init all image and label paths for every split, check to ensure they exist / are equal
        self.images, self.labels = {}, {}
        self.splits = splits
        for split in splits:
            self.images[split] = sorted(
                glob(str(self.root_dir / "images" / split / "**" / f"*.{extension}"), recursive=True)
            )
            self.labels[split] = sorted(
                glob(str(self.root_dir / "labels" / split / "**" / "*.txt"), recursive=True)
            )
            
            img_path = self.images[split]
            lbl_path = self.labels[split]
            len_image, len_lbl = len(img_path), len(lbl_path)
            if len_image != len_lbl:
                warnings.warn(f"{split} Image ({len_image}) and Label ({len_lbl}) directories have unequal elements.")
            if not img_path or not lbl_path:
                warnings.warn(f"{split} Image path exists: ({bool(img_path)}). Label path exists ({bool(lbl_path)})")
        
        self.gray = gray
        self.clahe = clahe
        self.current_split = splits[0]

    def __len__(self):
        return len(self.images[self.current_split])
    
    def set_split(self, split: str = "train"):
        """ Switches the active dataset split if legal """
        if split not in self.splits:
            warnings.warn("Illegal split used.")
            return
        self.current_split = split
    
    def __getitem__(self, idx: int):
        img_path = self.images[self.current_split][idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # openCV loads bgr by default

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
    
    def get_image_paths(self) -> list[str]:
        """ Returns a list of image paths """
        return self.images[self.current_split]