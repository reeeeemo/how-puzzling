import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
# from transformers import Sam2Processor, Sam2Model # if we use SAM 2
from pathlib import Path
from dataset import PuzzleDataset
import numpy as np
import matplotlib
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def segment_images_prompt(model,
                        processor,
                        images: list[Image.Image],
                        prompt: str) -> list:
    """
        Segment images given a list of images and a text prompt
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
            proc_out = processor(images=[img], text=prompt, return_tensors="pt")
            proc_out = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in proc_out.items()}
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
                    masks = masks.squeeze(1) # (N, 1, H, W) -> (N, H, W)
                elif masks.dim() == 2:
                    masks = masks.unsqueeze(0) # (H, W) -> (1, H, W)
                results.append(masks)
            else: # we got no masks :p
                w, h = img.size
                results.append(torch.empty((0, h, w), dtype=torch.bool))
    return results

def segment_images_bbox(model, 
                        processor, 
                        images: list[Image.Image], 
                        labels: list[list[torch.Tensor]]) -> list:
    """
        Segment images given a list of images and bounding boxes
        Args:
            model: SAM model
            processor: SAM processor
            images: list of PIL images to segment
            labels: list of bounding boxes aligned with image names
        Returns:
            list: masks for every image segmented
    """

    # convert yolo to pixel coords
    boxes = []
    for img, img_lbls in zip(images, labels):
        img_w, img_h = img.size
        img_boxes = []
        
        for lbl in img_lbls:
            _, cx, cy, w, h = lbl.tolist() # don't need class
            x_min = int((cx - w/2) * img_w)
            y_min = int((cy - h/2) * img_h)
            x_max = int((cx + w/2) * img_w)
            y_max = int((cy + h/2) * img_h)
            img_boxes.append([x_min, y_min, x_max, y_max])
        boxes.append(img_boxes) #  [x_min, y_min, x_max, y_max]

    # segment every image using bounding boxes
    # batching caused OOM issues :/
    results = []
    with torch.no_grad():
        for img, img_boxes in zip(images, boxes):
            # input through processor then move to device and compute output
            proc_out = processor(images=[img], input_boxes=[img_boxes], return_tensors="pt")
            proc_out = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in proc_out.items()}
            outputs = model(**proc_out, multimask_output=False)

            # resize back to original image size
            target_sizes = proc_out["original_sizes"]
            if isinstance(target_sizes, torch.Tensor):
                target_sizes = target_sizes.cpu()
            masks = processor.post_process_masks(outputs.pred_masks.cpu(), target_sizes)[0]

            # normalize to (N, H, W)
            if masks.dim() == 4:
                masks = masks.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
            elif masks.dim() == 2:
                masks = masks.unsqueeze(0)  # (H, W) -> (1, H, W)

            results.append(masks)
    return results

def overlay_masks(image: Image.Image, masks: list) -> Image.Image:
    """
        Given an image and a list of masks, overlay masks onto image
        Args:
            image: base image
            masks: masks to overlay
        Returns:
            Image: image with masks overlayed
    """
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha= mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def clean_masks(masks: list, img_shape: tuple[int, int]) -> torch.Tensor:
    """
        Given a list of masks and image shape, clean noise and get largest segmentation if multiple in same box
        Args:
            masks: list of masks
            img_shape: tuple of (height, width)
        Returns:
            torch.Tensor: tensor of cleaned masks
    """
    cleaned = []

    for mask in masks:
        mask_uint8 = (mask.cpu().numpy() * 255).astype(np.uint8)

        # clean noise 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea) # get largest contour

            clean_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(clean_mask, [largest], 0, 255, -1)
            cleaned.append(torch.from_numpy(clean_mask > 0))
            
    return torch.stack(cleaned) if cleaned else torch.empty((0, *img_shape), dtype=torch.bool)

def main():
    # tfloat32 for ampere gpus
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # SAM 3 model
    model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    # model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-base-plus").to(DEVICE)
    # processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-base-plus")

    cwd = Path("") / "data" / "original_puzzle" # downloaded data name
    dataset = PuzzleDataset(cwd) # no transforms, just want the data
    dataset_list = [(img, lbl) for img, lbl in dataset]
    images = []
    # labels = []

    for image, _ in dataset_list:
        images.append(image)
        # labels.append(label) # reword _ to label

    # segment_images_bbox for bounding box
    all_masks = segment_images_prompt(model, processor, images, "puzzle")

    for i, masks in enumerate(all_masks):
        # clean masks then overlay onto base image
        w, h = images[i].size
        clean = clean_masks(masks, (h, w))
        overlaid = overlay_masks(images[i], clean)
        img_cv = cv2.cvtColor(np.array(overlaid), cv2.COLOR_RGBA2BGR)

        cv2.imshow(f"Image {i}: {len(clean)} objects found", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()