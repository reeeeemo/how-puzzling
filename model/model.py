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
            List of raw segmentations
            Tuple of cosine similarities of each detected object,
            List of dicts tracking piece, side, and crop for every image.
        """
        results = self.model(imgs, verbose=False)
        edge_metadata = self.extract_all_edges(results, edge_width=15)
        similarities = self.compute_similarities(
            [data.get("crop") for data in edge_metadata]
        )
        return results, similarities, edge_metadata

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
        for image_i, result in enumerate(results):
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

                pts_i = np.rint(pts).astype(np.int32)
                mask_piece = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_piece, [pts_i], 255)

                all_pts = self.get_side_approx(pts, sides)
                centroid = self.get_centroid(pts)

                # crop
                for side_name, pts_tuple in all_pts.items():
                    if not pts_tuple or len(pts_tuple) < 2:
                        continue

                    pts_side = [pt for pt, _ in pts_tuple]
                    # radials = [r for _, r in pts_tuple]
                    cur_type = self.classify_edge_type(
                        points=pts_side,
                        centroid=centroid,
                        side=side_name,
                        epsilon_flat=30
                    )

                    if cur_type == "flat":
                        continue
                    elif cur_type == "knob":
                        color = (255, 0, 0)
                    else:
                        color = (0, 255, 0)

                    for pt in pts_side:
                        cv2.circle(
                            img,
                            (int(pt[0]), int(pt[1])),
                            2, color, 2
                        )

                    pts_side_i = np.array(pts_side, dtype=np.int32)
                    x_min = max(0, pts_side_i[:, 0].min() - edge_width)
                    x_max = min(w, pts_side_i[:, 0].max() + edge_width)
                    y_min = max(0, pts_side_i[:, 1].min() - edge_width)
                    y_max = min(h, pts_side_i[:, 1].max() + edge_width)
                    if x_max <= x_min or y_max <= y_min:
                        continue

                    # create mask then mask within piece mask to stay in piece
                    kernel = np.ones((7, 7), np.uint8)
                    mask_piece_closed = cv2.morphologyEx(
                        mask_piece,
                        cv2.MORPH_CLOSE,
                        kernel
                    )

                    mask_region = mask_piece_closed[y_min:y_max, x_min:x_max]
                    mask_region = cv2.morphologyEx(
                        mask_region,
                        cv2.MORPH_CLOSE, kernel
                    )

                    img_region = img[y_min:y_max, x_min:x_max]
                    cropped = cv2.bitwise_and(
                        img_region,
                        img_region,
                        mask=mask_region
                    )

                    edge_metadata.append({
                        "image_id": image_i,
                        "piece_id": piece_idx,
                        "side": side_name,
                        "crop": cropped,
                    })
                piece_idx += 1
        return edge_metadata

    def classify_piece(self,
                       edge_metadata: dict,
                       centroid: np.ndarray) -> tuple[str, dict]:
        """Classifies a puzzle piece based on its flat sides.

        Args:
            edge_metadata: dict of side, list of pts
            centroid: tuple of (x,y) pertaining to polygon center
        Returns:
            Tuple of:
             - Piece type (internal, corner, side_{side_type})
             - All side types (flat, knob, hole)
        """
        sides = {}
        for side, pts in edge_metadata.items():
            all_pts = [pt for pt, _ in pts]
            cur_type = self.classify_edge_type(
                points=all_pts,
                centroid=centroid,
                side=side,
                epsilon_flat=30
            )
            sides[side] = cur_type

        flat_sides = [s for s, typ in sides.items()
                      if typ == "flat"]
        flat_count = len(flat_sides)
        # can only have 0-2 flats, 2 and 0 are defined while 1 is ambigous
        if flat_count == 0:
            return "internal", sides
        if flat_count == 1:
            return f"side_{flat_sides[0]}", sides
        if flat_count == 2:
            sorted_flats = sorted(flat_sides)
            return f"corner_{sorted_flats[0]}_{sorted_flats[1]}", sides
        return "unknown", sides

    def get_centroid(self, points: np.ndarray, binary_mask: bool = False):
        """Compute centroid using moments.

        Args:
            points: polygon points to get centroid for
            binary_mask: whether we are computing centroid of bin mask or poly
        Returns:
            tuple of x, y for centroid
        """
        if not binary_mask:
            pts_i = np.rint(points).astype(np.int32)
            mu = cv2.moments(pts_i.reshape(-1, 1, 2))
        else:
            mu = cv2.moments(points)

        if mu.get("m00", 0) == 0:
            return None
        centroid = np.array(
            [
                mu["m10"] / mu["m00"],
                mu["m01"] / mu["m00"]
            ],
            dtype=np.float64
        )
        return centroid

    def get_side_approx(self, points, sides: dict):
        """Get all side approximations of a polygon.

        Args:
            points: polygon points to approximate
            sides: dict of sides + unit vectors that can be approximated
        Returns:
            dict of a points list for every side
        """
        centroid = self.get_centroid(points)

        new_pts = np.vstack([points[-1:], points, points[:1]])  # cyclic
        all_pts = {k: [] for k in sides.keys()}

        for cur_pt in new_pts:
            radial = cur_pt - centroid  # radial vec
            rmag = np.linalg.norm(radial)
            if rmag < 1e-12:
                continue
            radial /= rmag

            # given all sides find the greatest degree
            # use radial for global relativity
            scores = {
                name: float(np.dot(radial, np.asarray(side, dtype=np.float64)))
                for name, side in sides.items()
            }

            # if we have a dominant side (beats 30% of second place)
            best_side = max(scores, key=scores.get)
            best_score = scores[best_side]
            second_best = sorted(scores.values(), reverse=True)[1]

            if best_score > 0.5 and best_score > second_best * 1.3:
                all_pts[best_side].append((cur_pt, radial))

        return all_pts

    def classify_edge_type(
        self,
        points: np.ndarray,
        centroid: np.ndarray,
        side: str,
        epsilon_flat: int = 50,
        epsilon_curve: int = 100
         ):
        """Classifies edge from a baseline deviation.

        Args:
            points: numpy array of (x, y) tuples along an edge
            centroid: tuple of (x, y) centroid along the polygon
            side: current side of the edge being classified
            epsilon_flat: Threshold for deviation of a flat line
            k: Amt of points to consider for curve
        Returns:
            "knob", "hole", or "flat"
        """
        pts = np.asarray(points)
        # if side is vertical, check x, else y
        coord_idx = int(
            side in ["top", "bottom"]
        )

        # percentiles
        relevant_coords = pts[:, coord_idx]
        p10 = np.percentile(relevant_coords, 10)
        p90 = np.percentile(relevant_coords, 90)
        side_range = p90 - p10
        if side_range <= epsilon_flat:
            return "flat"

        mid_val = relevant_coords[len(relevant_coords) // 2]
        distance = np.abs(centroid[coord_idx] - mid_val)
        return "knob" if distance > epsilon_curve else "hole"

    # replacing this with classify_edge_type
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
