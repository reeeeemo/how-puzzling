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
    dataset_file = Path("..") / "dataset" / "data" / "gray_segmented_puzzle"
    images = glob(str(dataset_file / "images" / "train" / "**" / "*.jpg"), recursive=True)
    num_imgs = len(images)
    num_exists = sum(os.path.exists(image) for image in images)
    assert (num_imgs == num_exists), "Error: invalid image path"

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
    
def grayscale_clahe(img_path, output_path):
    """
        Applies grayscale + CLAHE to an image
        Args:
            img: image to apply filters then return    
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    enhanced = np.clip(clahe.apply(gray) + 30, 0, 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) # stack to 3 channel input
    cv2.imwrite(output_path, enhanced_bgr)

if __name__ == "__main__":
    project = Path("") / "model_outputs"
    project.mkdir(parents=True, exist_ok=True)
    
    # update images to bgr
    #dataset_file = Path("..") / "dataset" / "data" / "segmented_puzzle"
    #images = glob(str(dataset_file / "images" / "train" / "**" / "*.jpg"), recursive=True)
    #print(len(images))
    
    
    #new_ds_file = Path("..") / "dataset" / "data" / "gray_segmented_puzzle"
    #new_ds_images = new_ds_file / "images" / "train"
    #new_ds_labels = new_ds_file / "labels" / "train"
    #new_ds_images.mkdir(parents=True, exist_ok=True)
    #new_ds_labels.mkdir(parents=True, exist_ok=True)
    
    #for img in images:
    #    grayscale_clahe(img, f"{str(new_ds_images / Path(img).stem)}.jpg")

    train_yolo_model("yolo11n-seg.pt", 
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
                     name="train",
                     project=str(project)) 