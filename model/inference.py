from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from dataset.dataset import PuzzleDataset


def main():
    model_dir = Path(__file__).resolve().parent
    model = YOLO(model_dir / "model_outputs" / "train3" /"weights" / "best.pt")
    project_path = Path(__file__).resolve().parent.parent
    root_dir = project_path / "dataset" / "data" / "jigsaw_puzzle"
    dataset = PuzzleDataset(root_dir=root_dir, splits=["test"], extension="jpg")
    
    all_test_images = [img for img, _ in dataset]
    for image in all_test_images:
        new_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(image)[0]
        
        for c in results.masks.xy:
            contour_np = np.array(c, dtype=np.int32)
            cv2.drawContours(new_img, [contour_np], -1, (0,255,0), thickness=3)
            
        cv2.imshow("yolo output (cleaned)", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            

if __name__ == "__main__":
    main()