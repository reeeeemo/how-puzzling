import torch
import cv2
import numpy as np


def clean_masks(masks: list, img_shape: tuple[int, int]) -> torch.Tensor:
    """Given a list of masks and image shape, clean noise and
    get largest segmentation if multiple in same box.

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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_clean,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)  # get largest contour

            clean_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(clean_mask, [largest], 0, 255, -1)
            cleaned.append(torch.from_numpy(clean_mask > 0))

    return torch.stack(cleaned) if cleaned else torch.empty((0, *img_shape),
                                                            dtype=torch.bool)
