from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model.model import PuzzleImageModel
from dataset.dataset import PuzzleDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_similarites(project_path: Path):
    model_path = project_path / "model" / "model_outputs" / "train3" / "weights" / "best.pt"
    dataset_path = project_path / "dataset" / "data" / "jigsaw_puzzle"
    model = PuzzleImageModel(model_name=str(model_path), device=DEVICE)
    dataset = PuzzleDataset(root_dir=dataset_path, splits=["test"], extension="jpg")
    
    all_test_images = [
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for img, _ in dataset
    ]
    similarities = model(all_test_images)
    return similarities

def main():
    project_path = Path(__file__).resolve().parent
    output = project_path / "output"
    output.mkdir(exist_ok=True, parents=True)
    sims = compute_similarites(project_path)
    print(sims)
    
    #plt.imshow(sims, cmap="viridis")
    #plt.title("Piece-to-piece cosine similarity")
    #plt.savefig(str(output / "sim_heatmap.png"), dpi=300)
    #plt.close()
    
if __name__ == "__main__":
    main()