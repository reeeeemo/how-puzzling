from ultralytics import YOLO
import torch
from pathlib import Path
from glob import glob
import os 
import cv2
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_optimization():
    """
        Set some optimization variables
    """
    torch.set_float32_matmul_precision("high") # matmul precision
    torch.backends.cudnn.allow_tf32 = True # precision of convolution operations

def train_yolo_model(model_type: str, **kwargs):
    """
        Trains a YOLO model on a generated dataset 
        Args:
            model_type: string of segmentation model to run
            **kwargs: additional YOLO train settings, passed to model.train()
    """
    # get all images from dataset, confirm they exist
    splits = ["train", "val"]
    project_path = Path(__file__).resolve().parent.parent
    dataset_file = project_path / "dataset" / "data" / "jigsaw_puzzle"
    
    for split in splits:
        images = glob(str(dataset_file / "images" / split / "**" / "*.jpg"), recursive=True)
        num_imgs = len(images)
        num_exists = sum(os.path.exists(image) for image in images)
        assert (num_imgs == num_exists), f"Error: invalid {split} image path"


    # get yolo model
    model = YOLO(model_type).to(DEVICE)
    print("Model task:", model.task)

    # get yaml then start training
    yaml_file = str(dataset_file / "data.yaml")
    try:
        model.train(
            data=yaml_file,
            **kwargs
        )
    except Exception as e:
        print(f"YOLO train error: {e}")
        return
    
if __name__ == "__main__":
    this_dir = Path(__file__).resolve().parent 
    project = this_dir / "model_outputs"
    project.mkdir(parents=True, exist_ok=True)
    
    train_yolo_model(f"{str(this_dir)}/yolo11n-seg.pt", 
                     epochs=100,
                     patience=20, # early stopping
                     batch=8, # might have to adjust this for diff gpus
                     save_period=10,
                     optimizer="auto",
                     seed=42,
                     cos_lr=True, # cosine learning rate scheduler
                     plots=True, # save all plots
                     verbose=False,
                     cache=False,
                     exist_ok=True,
                     single_cls=True, # only one class being trained
                     name="train3",
                     project=str(project)) 