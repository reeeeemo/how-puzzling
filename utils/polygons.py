from model.model import PuzzleImageModel
import numpy as np
import cv2


def get_polygon_sides(poly, bbox, model: PuzzleImageModel) -> dict:

    pts = np.asarray(poly, dtype=np.float32)
    dense_pts = model.densify_polygons(pts, step=1)

    x1, y1, x2, y2 = map(int, bbox)
    pts_cropped = dense_pts - np.array([x1, y1])
    bbox_cropped = [0, 0, x2 - x1, y2 - y1]
    return model.get_side_approx(
        points=pts_cropped, bbox=bbox_cropped
    )


def create_binary_mask(poly, box, img_shape: tuple, image=None):
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
        return new_crop
    else:
        b_mask = np.zeros(img_shape, np.uint8)
        cv2.fillPoly(b_mask, [pts_i], 255)

        x1, y1, x2, y2 = map(int, box)
        b_crop = b_mask[y1:y2, x1:x2]

        return b_crop
