from glob import glob
from pathlib import Path
import cv2
import torch
import matplotlib.pyplot as plt

from model import PuzzleImageModel



def output_images(cwd: Path, output_cwd: Path, model: PuzzleImageModel):
    """
        Take in sample images and output their cropped version
    """
    images = glob(str(cwd / "test" / "**" / "*.jpg"), recursive=True)
    pair_results = model(images)
    
    for result, img_path in zip(pair_results, images):
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        
        stem = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}, skipping")
            continue
        
        h, w = img.shape[:2]
        
        for i, (box, cls_val) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, box)
            
            x1 = max(0, min(w-1,x1))
            x2 = max(0, min(w,x2))
            y1 = max(0,min(h-1,y1))
            y2 = max(0,min(h,y2))
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = img[y1:y2, x1:x2]
            try:
                cls_int = int(cls_val.item())
            except Exception:
                cls_int = int(float(cls_val))
                
            out_path = output_cwd / f"{stem}_crop_{i}_cls{cls_int}.jpg"
            cv2.imwrite(str(out_path), crop)
        

def extract_features_images(output_cwd: Path, model: PuzzleImageModel):
    """
        Uses feature encoder from YOLO model to get tensor of features from every cropped img
    """
    images = glob(str(output_cwd / "**" / "*.jpg"), recursive=True)
    
    results = model.forward_until_layer(images, -2) # before detect layer
    print(results.shape)
    return results

def global_average_pool_images(features):
    """
        Creates a vector of features using the global average pooling algorithm
    """
    return torch.mean(features, dim=[2,3]) # mean over width/height


def compute_cosine_sim_images(pools):
    """
        computes cosine similarity across each image using features
    """
    pooled = torch.nn.functional.normalize(pools, dim=1)
    sims = pooled @ pooled.T # cosine sim
    return sims

if __name__ == "__main__":
    cwd = Path("")
    output = cwd / "output"
    output.mkdir(exist_ok=True, parents=True)
    
    model_name = "best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PuzzleImageModel(model_name, device)
    
    feats = extract_features_images(output, model)
    pools = global_average_pool_images(feats)
    
    sims = compute_cosine_sim_images(pools)
    
    sims_np = sims.cpu().numpy()
    plt.imshow(sims_np, cmap="viridis")
    plt.title("Piece-To-Piece Cosine Similarity")
    plt.savefig("similarity_heatmap.png", dpi=300)
    plt.close()    