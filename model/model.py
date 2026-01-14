from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel


class PuzzleImageModel(nn.Module):
    """Model that segments all individual jigsaw puzzle pieces in an image.

    Computes cosine similarity between each piece's edges.
    Attributes:
        model: pretrained YOLO model to segment images with
        device: device to run matmul / models on
        similarity_processor: DINOv3 image encoder's processor
        similarity_model: DINOv3 image encoder
    """
    def __init__(self, model_name: str = "best.pt", device: str = "cpu"):
        super().__init__()
        torch.set_float32_matmul_precision("high")  # optimize matmuls

        # download from (https://huggingface.co/reeeemo/puzzle-segment-model)
        self.model = YOLO(model_name).to(device)
        self.device = device

        # dinov3 requires gated access
        self.similarity_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        self.similarity_model = AutoModel.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )

    def forward(self, imgs) -> tuple[
                                     torch.Tensor,
                                     dict[int, list],
                                     list[dict]
                                    ]:
        """Segment and compute edge-to-edge similarity on all puzzle images.

        Args:
            imgs: list of numpy arrays
        Returns:
            Tuple of cosine similarities of each detected object,
            Xyxy coords of boxes cropped relative to segmented mask,
            List of dicts tracking piece, side, and crop for every image.
        """
        results = self.model(imgs, verbose=False)
        edge_metadata = self.extract_all_edges(results, edge_width=80)
        boxes_per_image = self.crop_images(results, imgs)
        similarities = self.compute_similarities(
            [data.get("crop") for data in edge_metadata]
        )
        return similarities, boxes_per_image, edge_metadata

    def extract_all_edges(self, results, edge_width: int = 60) -> dict:
        """Extract each edge crop for all pieces.

        Args:
            results: YOLO segmentation results
        Returns:
            list of dicts of piece idx, side type, and crop
        """
        edge_metadata = []

        sides = {
            "bottom": (0, 1),
            "top": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        for result in results:  # all image segmentation results
            img = result.orig_img
            h, w = img.shape[:2]

            piece_idx = 0
            for poly in getattr(result.masks, "xy", []):
                pts = np.asarray(poly, dtype=np.float32)
                if pts.size == 0:
                    continue

                pts = self.densify_polygons(pts, step=1.0)
                pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h-1)

                # piece mask for crops
                pts_i = np.rint(pts).astype(np.int32)
                mask_piece = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_piece, [pts_i], 255)

                # compute centroid using moments
                mu = cv2.moments(pts_i.reshape(-1, 1, 2))
                if mu.get("m00", 0) == 0:
                    piece_idx += 1
                    continue
                centroid = np.array(
                    [
                        mu["m10"] / mu["m00"],
                        mu["m01"] / mu["m00"]
                    ],
                    dtype=np.float64
                )

                new_pts = np.vstack([pts[-1:], pts, pts[:1]])  # cyclic
                all_pts = {k: [] for k in sides.keys()}

                for cur_pt in new_pts:
                    radial = cur_pt - centroid  # radial vec
                    rmag = np.linalg.norm(radial)
                    if rmag < 1e-12:
                        continue
                    radial /= rmag

                    # given all sides find the greatest degree
                    # use radial for global relativity
                    best_side, _ = max(
                        ((
                          name,
                          float(np.dot(radial,
                                       np.asarray(side, dtype=np.float64)))
                          ) for name, side in sides.items()
                         ),
                        key=lambda t: t[1]
                    )
                    all_pts[best_side].append((cur_pt, radial))

                # this is where the crop is done
                for side_name, pts_tuple in all_pts.items():
                    if not pts_tuple or len(pts_tuple) < 2:
                        continue

                    pts_side = [pt for pt, _ in pts_tuple]
                    radials = [r for _, r in pts_tuple]

                    if self.is_flat_side(pts_side,
                                         epsilon=25,
                                         vertical=(
                                             side_name not in ["top", "bottom"]
                                             )
                                         ):
                        continue

                    inner_points = []

                    for i in range(len(pts_side)):
                        prev_i = (i-1) % len(pts_side)
                        next_i = (i+1) % len(pts_side)

                        tangent = pts_side[next_i] - pts_side[prev_i]
                        tmag = np.linalg.norm(tangent)

                        if tmag < 1e-6:
                            perp = -radials[i]  # rough inward point calc
                        else:
                            tangent /= tmag
                            # 90 degree rotation then flip signs if outward
                            perp = np.array([-tangent[1], tangent[0]])
                            if np.dot(perp, (centroid-pts_side[i])) < 0:
                                perp = -perp

                        inner_points.append(pts_side[i]+perp*(edge_width//2))

                    strip_pts = np.array(pts_side + inner_points[::-1],
                                         dtype=np.int32)

                    # get bbox + clamp to image boundaries
                    x_min, y_min = np.min(strip_pts, axis=0).astype(int)
                    x_max, y_max = np.max(strip_pts, axis=0).astype(int)

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)

                    if x_max <= x_min or y_max <= y_min:
                        continue

                    # create mask then mask within piece mask to stay in piece
                    kernel = np.ones((7, 7), np.uint8)
                    mask_piece_closed = cv2.morphologyEx(mask_piece,
                                                         cv2.MORPH_CLOSE,
                                                         kernel)

                    # fix polygon artifacts
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [strip_pts], 255)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.bitwise_and(mask, mask_piece_closed)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                    # dilate and erode for any extra artifacts
                    mask = cv2.dilate(mask, kernel, iterations=2)
                    mask = cv2.erode(mask, kernel, iterations=1)

                    # extract crop
                    result_img = cv2.bitwise_and(img, img, mask=mask)
                    cropped = result_img[y_min:y_max, x_min:x_max]
                    mask_crop = mask[y_min:y_max, x_min:x_max]  # cleaner edges

                    cropped = cv2.bitwise_and(cropped, cropped, mask=mask_crop)

                    edge_metadata.append({
                        "piece_id": piece_idx,
                        "side": side_name,
                        "crop": cropped,
                    })
                piece_idx += 1

        return edge_metadata

    def is_flat_side(self, points, epsilon: int = 1, vertical: bool = False):
        """Return True if side is flat, else false

        Args:
            points: numpy array of (x,y) tuples
            epsilon: Threshold for normalized max deviation
            vertical: whether to compare y or x deviation
        """
        coord = 1 - int(vertical)
        np_pts = np.array(points)

        min_val = np.min(np_pts[:, coord])
        max_val = np.max(np_pts[:, coord])
        side_range = max_val - min_val

        return side_range <= epsilon

    def densify_polygons(self, pts, step: int = 1.0):
        """Walk over each edge of polygon and add points.

        Args:
            pts: sparse polygon points
            step (int): number of pts to add over an edge
        Returns:
            numpy array of all points representing a polygon
        """
        dense = []

        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i+1) % len(pts)]  # nxt point or 0

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

    def compute_embeddings(self, imgs):
        """Compute embeddings of an image using the encoder.
        Args:
            imgs: list of numpy arrays
        Returns:
            dict of results from encoder.
        """
        with torch.no_grad():
            inputs = self.similarity_processor(
                images=imgs,
                return_tensors="pt"
            ).to(self.device)

            inputs = inputs.to(self.device)
            outputs = self.similarity_model(**inputs)

        return outputs

    def compute_similarities(self, imgs):
        """Compute a similarity matrix of multiple images using cosine
        similarity.

        Args:
            imgs: list of numpy arrays
        Returns:
            Tensor of img-to-img cosine similarities
        """
        outputs = self.compute_embeddings(imgs)
        # dinov3 has overall cls embedding
        image_feats = outputs.last_hidden_state[:, 0, :]
        image_feats = nn.functional.normalize(image_feats, dim=1)

        # cos sim is (a \dot b) / ||a|| x ||b||
        # we normalize so magnitude is 1 which reduces to a \dot b
        sim_mat = torch.mm(image_feats, image_feats.T)
        sim_mat = torch.clamp(sim_mat, 0.0, 1.0)
        sim_mat.fill_diagonal_(1.0)
        return sim_mat

    def crop_images(self, results, imgs):
        """Crop the images per segmentation

        Args:
            results: segmentation masks from model
            imgs: list of numpy arrays
        Returns:
            tuple: tensor of batched cropped,
            xyxy coords of cropped box
        """
        boxes_per_image = {i: [] for i in range(len(imgs))}

        for idx, (result, img) in enumerate(zip(results, imgs)):
            boxes = result.boxes.xyxy
            h, w = img.shape[:2]

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(w-1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h-1, y1))
                y2 = max(0, min(h, y2))

                if x2 <= x1 or y2 <= y1:
                    continue  # boxes out of bounds

                boxes_per_image[idx].append((x1, y1, x2, y2))

        return boxes_per_image
