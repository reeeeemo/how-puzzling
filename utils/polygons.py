from model.model import PuzzleImageModel
import numpy as np
import cv2


def get_polygon_sides(poly: np.ndarray,
                      bbox: list,
                      model: PuzzleImageModel,
                      resize_ratio: float = 1.0) -> dict:
    """Creates a dict of side: np.array[points] for all polygon pts

    Args:
        poly: list of points representing a polygon
        bbox: list of points representing a bounding box
        model: PuzzleImageModel to call side approximation functions
        resize_ratio: % of w/h to resize for points
    Returns:
        dict: dict of side: pts for all cardinal sides
    """
    pts = np.asarray(poly, dtype=np.float32)
    dense_pts = model.densify_polygons(pts, step=1)

    x1, y1, x2, y2 = map(int, bbox)
    pts_cropped = (dense_pts - np.array([x1, y1])) * resize_ratio
    bbox_cropped = [0, 0, (x2 - x1) * resize_ratio, (y2 - y1) * resize_ratio]
    return model.get_side_approx(
        points=pts_cropped, bbox=bbox_cropped
    )


def create_binary_mask(poly: np.ndarray,
                       box: list,
                       img_shape: tuple,
                       image: np.ndarray = None,
                       resize_ratio: float = 1.0) -> np.ndarray:
    """Creates a grayscsale binary mask of a list of polygon points.

    If image is provided, provide the mask in RGB.
    Args:
        poly: list of points representing a polygon
        box: list of points representing a bounding box
        img_shape: shape of target image
        image: image to provide for RGB
        resize_ratio: % of w/h to resize for image
    Returns:
        np.ndarray: grayscale or RGB mask
    """
    pts = np.asarray(poly, dtype=np.float32)
    if pts.size == 0:
        return

    pts_i = np.rint(pts).astype(np.int32)

    if image is not None and len(image.shape) == 3:
        b_mask = np.zeros_like(image)
        cv2.fillPoly(b_mask, [pts_i], (255, 255, 255))

        x1, y1, x2, y2 = map(int, box)

        mask_crop = b_mask[y1:y2, x1:x2]
        img_crop = image[y1:y2, x1:x2]

        new_crop = cv2.bitwise_and(mask_crop, img_crop)
    else:
        b_mask = np.zeros(img_shape, np.uint8)
        cv2.fillPoly(b_mask, [pts_i], 255)

        x1, y1, x2, y2 = map(int, box)
        new_crop = b_mask[y1:y2, x1:x2]

    ref_h = int(new_crop.shape[0] * resize_ratio)
    ref_w = int(new_crop.shape[1] * resize_ratio)
    new_crop = cv2.resize(new_crop, (ref_w, ref_h))
    return new_crop
