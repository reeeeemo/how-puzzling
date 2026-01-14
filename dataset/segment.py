import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from pathlib import Path
from dataset.dataset import PuzzleDataset
import numpy as np
import cv2
from utils.masks import clean_masks
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# dataset generated: https://huggingface.co/datasets/reeeemo/jigsaw_puzzle.
# you can also add new puzzles to the dataset as well (no need for labels)

# Generates a dataset given a YOLO-style format of images

# note that <your_filename_here> has to be in dataset/data/*
# Use (root): python -m dataset.segment --filename <your_filename_here>
# Example: python -m dataset.segment --filename jigsaw_puzzle

def segment_images_prompt(model,
                          processor,
                          images: list[Image.Image],
                          prompt: str) -> list:
    """Segment images given a list of images and a text prompt.

    Args:
        model: SAM 3 model
        processor: SAM 3 processor
        images: list of PIL images to segment
        prompt: prompt for image segmentation
    Returns:
        list: masks for every image segmented
    """
    results = []
    with torch.no_grad():
        for img in images:
            # input through processor, move to device and compute segmentations
            proc_out = processor(images=[img],
                                 text=prompt,
                                 return_tensors="pt")
            proc_out = {
                k: (
                    v.to(DEVICE) if isinstance(v, torch.Tensor)
                    else v
                   )
                for k, v in proc_out.items()
            }
            outputs = model(**proc_out)

            # get original size then postprocess segmentations + resize
            target_sizes = proc_out.get("original_sizes")
            if isinstance(target_sizes, torch.Tensor):
                target_sizes = target_sizes.cpu().tolist()
            result = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=target_sizes
            )

            # normalize to (N, H, W)
            if result and "masks" in result[0]:
                masks = result[0]["masks"]
                if masks.dim() == 4:
                    masks = masks.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
                elif masks.dim() == 2:
                    masks = masks.unsqueeze(0)  # (H, W) -> (1, H, W)
                results.append(masks)
            else:  # we got no masks :p
                w, h = img.size
                results.append(torch.empty((0, h, w), dtype=torch.bool))
    return results


def mask_to_yolo_polygons(mask: np.ndarray, img_w: int, img_h: int):
    """Convert a binary mask to a list of YOLO-style normalized polygons.

    Args:
        mask: mask to convert
        img_w, img_h: image width and height
    Returns:
        polygons: list of polygons representing binary mask
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:  # not enough outline pieces
            continue
        # normalize to [0, 1]
        poly = [(x[0][0] / img_w, x[0][1] / img_h) for x in cnt]
        polygons.append(poly)
    return polygons


def polygons_to_yolo_lines(polygons, class_id: int = 0):
    """
        Flatten polygons into YOLO segmentation text lines
        Args:
            polygons: list of polygons representing a mask
            class_id: class ID of that mask
        Returns:
            lines: list of polygons formatted for YOLO training
    """
    lines = []
    for poly in polygons:
        flat = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
        lines.append(f"{class_id} {flat}")
    return lines


def segment_all_images(file_name: str = "original_puzzle"):
    """Segment all images inside of the downloaded YOLO-style dataset.

    Save to a new generated dataset.
    """
    # tfloat32 for ampere gpus
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # SAM 3 model
    model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    splits = ["train", "val"]
    proj_dir = Path(__file__).resolve().parent
    cwd = proj_dir / "data" / file_name  # downloaded data name
    dataset = PuzzleDataset(root_dir=cwd,
                            gray=True,
                            clahe=True,
                            splits=splits)

    # create output data file (yolo style)
    output_dir = proj_dir / "data" / "segmented_puzzle"

    # get val / train data paths and compute images
    all_images, all_masks = {}, {}
    images_directories, labels_directories = {}, {}
    images_paths = {}
    for split in splits:
        dataset.set_split(split)
        # get all images preloaded from dataset class
        all_images[split] = [img for img, _ in dataset]
        images_paths[split] = dataset.get_image_paths()

        # segment with prompts, see dataset/README.md for details
        all_masks[split] = segment_images_prompt(model,
                                                 processor,
                                                 all_images[split],
                                                 "puzzle")

        images_directories[split] = output_dir / "images" / split
        images_directories[split].mkdir(parents=True, exist_ok=True)
        labels_directories[split] = output_dir / "labels" / split
        labels_directories[split].mkdir(parents=True, exist_ok=True)

    # masks size guaranteed same as image and label size
    for split in splits:
        for i, masks in enumerate(all_masks[split]):
            # clean masks
            w, h = all_images[split][i].size
            clean = clean_masks(masks, (h, w))
            new_img_path = (
                            images_directories[split] /
                            f"{Path(images_paths[split][i]).stem}.jpg"
                           )

            # reopen image without transforms
            clean_img = cv2.imread(images_paths[split][i])
            clean_img = cv2.resize(clean_img,
                                   (1920, 1080),
                                   interpolation=cv2.INTER_LANCZOS4)

            cv2.imwrite(str(new_img_path), clean_img)

            yolo_lines = []
            if clean.numel() == 0:
                continue  # skip empty tensors

            for j in range(clean.shape[0]):
                mask_np = clean[j].cpu().numpy().astype(np.uint8)
                polys = mask_to_yolo_polygons(mask_np, w, h)
                lines = polygons_to_yolo_lines(polys, class_id=0)
                yolo_lines.extend(lines)

            label_path = (
                          labels_directories[split] /
                          f"{Path(images_paths[split][i]).stem}.txt"
                         )
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dataset Generation via Segmentation"
    )
    parser.add_argument("--filename",
                        type=str,
                        required=True,
                        help="filename of dataset in dataset/data/*")
    args = parser.parse_args()
    segment_all_images(args.filename)
