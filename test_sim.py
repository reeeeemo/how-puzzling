from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.stats import rankdata

from model.model import PuzzleImageModel
from dataset.dataset import PuzzleDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute(model_path: Path, dataset_path: Path, split: str):
    """Get the similarity matrix of the requested split."""
    model = PuzzleImageModel(model_name=str(model_path), device=DEVICE)
    dataset = PuzzleDataset(root_dir=dataset_path, splits=[split], extension="jpg")
    all_test_images = [
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for img, _ in dataset
    ]

    results = model.model(all_test_images, verbose=False)
    return results

def densify_polygons(pts, step=1.0):
    """Walk over each edge of polygon and add points.
    
    Args:
        step (int): # of pts to add over an edge
    Returns:
        numpy array of all points representing a polygon.
    """
    dense = []
    
    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i+1)%len(pts)]  # nxt point or 0
        
        # get unit vector and decide how many pts to add
        v = p1 - p0
        length = np.linalg.norm(v)
        if length < 1e-6:
            continue
        
        direction = v / length
        n = int(length // step)
        
        for k in range(n+1):
            dense.append(p0+direction*k*step)
        dense.append(p1)
    return np.asarray(dense)

def main():
    project_path = Path(__file__).resolve().parent
    output = project_path / "output"
    model_path = project_path / "model" / "model_outputs" / "train3" / "weights" / "best.pt"
    dataset_path = project_path / "dataset" / "data" / "jigsaw_puzzle"
    output.mkdir(exist_ok=True, parents=True)

    results = compute(model_path, dataset_path, split="test")
    
    for result in results:
        img = result.orig_img
        h, w = img.shape[:2]
        display = img.copy()
        for ci, c in enumerate(result):
            polys = getattr(c.masks, "xy", [])
            sides = {
                "bottom": (0,1),
                "top": (0,-1),
                "left": (-1, 0),
                "right": (1, 0),
            }
            all_points = {}
            for poly in polys: # extract_side is called here
                pts = np.asarray(poly, dtype=np.float32)
                if pts.size == 0:
                    continue
                
                pts = densify_polygons(pts, step=1.0)
                pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h-1)
                
                # compute centroid using moments
                pts_i = np.rint(pts).astype(np.int32)
                mu = cv2.moments(pts_i.reshape(-1,1,2))
                centroid = np.array(
                    [
                    mu["m10"] / mu["m00"],
                    mu["m01"] / mu["m00"]
                    ],
                    dtype=np.float64
                )
                
                # cyclic then find normals
                new_pts = np.vstack([pts[-1:], pts, pts[:1]])
                
                #for i in range(len(new_pts)-2):
                for cur_pt in new_pts:
                    print(cur_pt)
                    # find distance from centroid to boundary point
                    radial = cur_pt - centroid
                    rmag = np.linalg.norm(radial)
                    if rmag < 1e-12:
                        continue
                    radial /= rmag
                    
                    # given all sides find the greatest degree
                    # use radial for global relativety instead of normal (which is local)
                    best_side, _ = max(
                        ((name, float(np.dot(radial, np.asarray(side, dtype=np.float64)))) for name, side in sides.items()),
                        key=lambda t: t[1]
                    )
                    all_points.setdefault(best_side, []).append(cur_pt)

                    # rudimentary way for diff sides
                    if best_side == "left":
                        color = (0,255,0)
                    elif best_side == "right":
                        color = (255, 0, 0)
                    elif best_side == "top":
                        color = (0, 0, 255)
                    else: # bottoms
                        color = (255, 255, 255)
                    p_draw = tuple(np.rint(cur_pt).astype(int))
                    cv2.circle(display, p_draw, 1, color, 1)
                
        cv2.imshow("mask overlay", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
if __name__=="__main__":
    main()