from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from dataset.dataset import PuzzleDataset
import argparse

# Note that <model_path> has to be in model/*

# Runs a sample inference on a specific dataset split with a segmentation model

# Use (root):
# python -m model.inference
#   --dataset <dataset_path>
#   --model <model_path>
#   --split <split>
# Example:
# python -m model.inference
#   --dataset dataset/data/jigsaw_puzzle
#   --model puzzle-segment-model/best.pt
#   --split test


def inference(dataset_name: str, model_name: str, split: str):
    model_dir = Path(__file__).resolve().parent
    model = YOLO(model_dir / model_name)
    project_path = Path(__file__).resolve().parent.parent
    root_dir = project_path / dataset_name
    dataset = PuzzleDataset(root_dir=root_dir,
                            splits=[split],
                            extension="jpg")

    all_test_images = [img for img, _ in dataset]
    for image in all_test_images:
        new_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(image)[0]

        for c in results.masks.xy:
            contour_np = np.array(c, dtype=np.int32)

            cv2.drawContours(new_img,
                             [contour_np],
                             -1,
                             (0, 255, 0),
                             thickness=3)

        cv2.imshow("yolo output", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Inference of a dataset split"
    )
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="filepath of dataset")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="filepath of model")
    parser.add_argument("--split",
                        type=str,
                        required=True,
                        help="split to inference")
    args = parser.parse_args()
    inference(args.dataset, args.model, args.split)
