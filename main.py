from glob import glob
from pathlib import Path
import threading
import queue
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import PuzzleImageModel


if __name__ == "__main__":
    cwd = Path("")
    output = cwd / "output"
    output.mkdir(exist_ok=True, parents=True)
    
    model_name = str(cwd / "models" / "best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PuzzleImageModel(model_name, device)
    
    # get image paths (temp until we use cv2)
    img_paths = glob(str(cwd / "images" / "**" / "*.jpg"), recursive=True)
    # normalize and resize
    images = []
    for img_path in img_paths:
        image = cv2.imread(img_path)[:, :, ::-1] 
        image = cv2.resize(image, (640, 640)) 
        images.append(image)
        
    # run the vision framework on each image to get cosine sim
    sims_np = model(images)
    plt.imshow(sims_np, cmap="viridis")
    plt.title("Piece-To-Piece Cosine Similarity")
    plt.savefig("similarity_heatmap.png", dpi=300)
    plt.close()    