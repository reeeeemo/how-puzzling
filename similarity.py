from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.stats import rankdata

from model.model import PuzzleImageModel
from dataset.dataset import PuzzleDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_similarites(model_path: Path, dataset_path: Path, split: str):
    """
        Given a path to a dataset and model, 
        get the similarity matrix of the requested split
        Args:
            model_path: path to pretrained segmentation model
            dataset_path: path to dataset of images in YOLO format
            split: split to take images from
        Returns:
            tuple of cosine similarity matrix,
            all images from split that was segmented,
            xyxy coords of boxes cropped relative to segmented masks
    """

    model = PuzzleImageModel(model_name=str(model_path), device=DEVICE)
    dataset = PuzzleDataset(root_dir=dataset_path, splits=[split], extension="jpg")
    
    all_test_images = [
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for img, _ in dataset
    ]
    similarities, boxes_per_image = model(all_test_images)
    return similarities, all_test_images, boxes_per_image

def main():
    project_path = Path(__file__).resolve().parent
    output = project_path / "output"
    model_path = project_path / "model" / "model_outputs" / "train3" / "weights" / "best.pt"
    dataset_path = project_path / "dataset" / "data" / "jigsaw_puzzle"
    output.mkdir(exist_ok=True, parents=True)
     
    
    sims, images, boxes_per_image = compute_similarites(model_path, dataset_path, "test")
    
    for idx, boxes in boxes_per_image.items():        
        for i in range(len(sims)):
            img = images[idx].copy()
            sim_col = sims[:, i]
            
            # mask to exclude diagonal elements, then compute ranking
            mask = np.ones(len(sim_col), dtype=bool)
            mask[i] = False
            
            masked_sims = sim_col[mask]
            ranked_sim = rankdata(masked_sims, method="ordinal") - 1 # 0 indexxed
            normalized_ranks = ranked_sim / (len(ranked_sim) - 1) if len(ranked_sim) > 0 else ranked_sim
            
            # reconstruct with masked value, but cur piece == -1           
            full_ranks = np.zeros(len(sim_col))
            full_ranks[mask] = normalized_ranks
            full_ranks[i] = -1
            
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                
                if j == i: # the piece we're comparing
                    cv2.rectangle(img, (x1,y1), (x2,y2), (255, 255, 255), 2)
                else:
                    color = int(full_ranks[j] * 255)
                    cv2.rectangle(img, (x1, y1), (x2,y2), (0, color, 255-color), 2)
                
            cv2.imshow(f"similarity results for piece {i}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        
    return
    
    """
    for img_idx, piece_indices in pieces_by_image.items():
        img = images[img_idx].copy()
        boxes = boxes_per_image[img_idx]
        
        sub_sim = sims[np.ix_(piece_indices, piece_indices)]
        
        for local_idx, (piece_idx, box) in enumerate(zip(piece_indices, boxes)):
            x1, y1, x2, y2 = box
            
            avg_sim = sub_sim[local_idx].mean().item()
            color = int(avg_sim * 255)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, color, 255-color), 2)
            
        cv2.imshow(f"annotated_img_{img_idx}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
       """ 
    
if __name__ == "__main__":
    main()