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


def create_binary_mask(poly, box, img_shape: tuple):
    pts = np.asarray(poly, dtype=np.float32)
    if pts.size == 0:
        return

    pts_i = np.rint(pts).astype(np.int32)
    b_mask = np.zeros(img_shape, np.uint8)
    cv2.fillPoly(b_mask, [pts_i], 255)

    x1, y1, x2, y2 = map(int, box)
    b_crop = b_mask[y1:y2, x1:x2]

    return b_crop
