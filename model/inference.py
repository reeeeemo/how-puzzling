from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from utils.masks import clean_masks
from dataset.dataset import PuzzleDataset
import random
import torch

def main():
    model = YOLO(str(Path("model") / "model_outputs" / "train3" /"weights" / "best.pt"))
    project_path = Path(__file__).resolve().parent.parent
    root_dir = project_path / "dataset" / "data" / "jigsaw_puzzle"
    dataset = PuzzleDataset(root_dir=root_dir, splits=["test"], extension="jpg")
    
    all_test_images = [img for img, _ in dataset]
    for image in all_test_images:
        results = model(image)[0]
        cv2.imshow("yolo output", results.plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            

if __name__ == "__main__":
    main()