from ultralytics import YOLO
from pathlib import Path
from glob import glob
import cv2

def main():
    model = YOLO(str(Path("") / "model_outputs" / "train" /"weights" / "best.pt"))
    images = glob(str(Path("..") / "dataset" / "data" / "gray_segmented_puzzle" / "images" / "val" / "**" / "*.jpg"), recursive=True)
    for image in images:
        for r in model(image):
            frame = r.plot()
            cv2.imshow("yolo output", frame)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
            

if __name__ == "__main__":
    main()