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
            edge_width: amount of space to crop for each edge
        Returns:
            list of dicts of piece idx, side type, and crop
        """
        edge_metadata = []

        for image_i, result in enumerate(results):
            img = result.orig_img
            h, w = img.shape[:2]

            piece_idx = 0
            for i, poly in enumerate(getattr(result.masks, "xy", [])):
                pts = np.asarray(poly, dtype=np.float32)
                if pts.size == 0:
                    continue

                pts = self.densify_polygons(pts, step=1.0)
                pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h-1)

                pts_i = np.rint(pts).astype(np.int32)
                mask_piece = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_piece, [pts_i], 255)

                all_pts = self.get_side_approx(
                    points=pts_i,
                    bbox=result.boxes.xyxy[i]
                )
                centroid = self.get_centroid(pts)

                # crop
                for side_name, pts_tuple in all_pts.items():
                    if pts_tuple is None or len(pts_tuple) < 2:
                        continue

                    pts_side = np.array(pts_tuple)
                    cur_type = self.classify_edge_type(
                        points=pts_side,
                        centroid=centroid,
                        side=side_name,
                        epsilon_flat=30
                    )

                    if cur_type == "flat":
                        continue
                    # visualization code, keep for testing remove for prod
                    # elif cur_type == "knob":
                    #    color = (255, 0, 0)
                    # else:
                    #    color = (0, 255, 0)

                    # for pt in pts_side:
                    #    cv2.circle(
                    #        img,
                    #        (int(pt[0]), int(pt[1])),
                    #        2, color, 2
                    #    )

                    x_min = max(0, pts_side[:, 0].min() - edge_width)
                    x_max = min(w, pts_side[:, 0].max() + edge_width)
                    y_min = max(0, pts_side[:, 1].min() - edge_width)
                    y_max = min(h, pts_side[:, 1].max() + edge_width)
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
                        "side_type": cur_type,
                        "pts": pts_i,
                    })
                piece_idx += 1
        return edge_metadata

    def classify_piece(self,
                       edge_metadata: dict) -> tuple[str, dict]:
        """Classifies a puzzle piece based on its flat sides.

        Args:
            edge_metadata: dict of side, list of pts
        Returns:
            Tuple of:
             - Piece type (internal, corner, side_{side_type})
             - All side types (flat, knob, hole)
        """
        sides = {}
        for meta in edge_metadata:
            sides[meta["side"]] = meta["side_type"]

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

    def get_side_approx(self, points: np.ndarray, bbox: list):
        """Get all side approximations of a polygon.

        Args:
            points: polygon points to approximate
            bbox: bounding box of whole mask
        Returns:
            dict of a points list for every side
        """
        sides = ["left", "right", "top", "bottom"]
        all_pts = {k: [] for k in sides}
        pts_i = np.rint(points).astype(np.int32)

        # find 4 edges of polygon
        x1, y1, x2, y2 = map(int, bbox)
        side_edges = {
            "top_left": (x1, y1),
            "bottom_left": (x1, y2),
            "top_right": (x2, y1),
            "bottom_right": (x2, y2)
        }

        corners = {}
        picked_pts = set()
        for side, coord in side_edges.items():
            # get top 10 distances from bbox
            distances = []
            for pt in pts_i:
                new_dist = np.sqrt(
                    (coord[0] - pt[0]) ** 2 +
                    (coord[1] - pt[1]) ** 2
                )
                distances.append((new_dist, pt))
            distances.sort(key=lambda x: x[0])
            potential_edges = [pt for _, pt in distances[:10]]

            # get closest x to the edge
            coord_idx = int("top" not in side and "bottom" not in side)
            closest_pt = None
            closest_dist = float("inf")
            for edge_pt in potential_edges:
                if tuple(edge_pt) in picked_pts:
                    continue

                if "left" in side or "top" in side:
                    new_dist = edge_pt[coord_idx] - coord[coord_idx]
                else:
                    new_dist = coord[coord_idx] - edge_pt[coord_idx]

                if new_dist < closest_dist:
                    closest_pt = edge_pt
                    closest_dist = new_dist
            corners[side] = tuple(closest_pt)
            picked_pts.add(tuple(closest_pt))

        corner_indices = {}
        for corner_name, corner_pt in corners.items():
            for i, pt in enumerate(pts_i):
                if tuple(pt) == corner_pt:
                    corner_indices[corner_name] = i
                    break

        tl_idx = corner_indices["top_left"]
        tr_idx = corner_indices["top_right"]
        bl_idx = corner_indices["bottom_left"]
        br_idx = corner_indices["bottom_right"]

        indices = sorted([
            (tl_idx, "top_left"),
            (tr_idx, "top_right"),
            (br_idx, "bottom_right"),
            (bl_idx, "bottom_left")
        ])

        for i in range(len(indices)):
            start_idx, start_corner = indices[i]
            end_idx, end_corner = indices[(i+1) % len(indices)]

            corner_pair = {start_corner, end_corner}
            if corner_pair == {"top_left", "top_right"}:
                side_key = "top"
            elif corner_pair == {"top_right", "bottom_right"}:
                side_key = "right"
            elif corner_pair == {"bottom_right", "bottom_left"}:
                side_key = "bottom"
            elif corner_pair == {"bottom_left", "top_left"}:
                side_key = "left"
            else:
                continue

            if end_idx > start_idx:  # pts between start and end
                all_pts[side_key] = points[start_idx:end_idx+1]  # pts_i
            else:  # wrap points
                all_pts[side_key] = np.concatenate([
                    points[start_idx:],
                    points[:end_idx+1]
                ])

        for side_key in all_pts:
            if len(all_pts[side_key]) == 0:
                all_pts[side_key] = np.zeros((0, 2), dtype=np.int32)
            else:
                all_pts[side_key] = np.atleast_2d(all_pts[side_key])
        return all_pts

    def classify_edge_type(
        self,
        points: np.ndarray,
        centroid: np.ndarray,
        side: str,
        epsilon_flat: int = 50,
         ):
        """Classifies edge from a baseline deviation.

        Args:
            points: numpy array of (x, y) tuples along an edge
            centroid: tuple of (x, y) centroid along the polygon
            side: current side of the edge being classified
            epsilon_flat: Threshold for deviation of a flat line
        Returns:
            "knob", "hole", or "flat"
        """
        pts = points
        coord_idx = int(
            side in ["top", "bottom"]
        )
        perp_coord_idx = coord_idx - 1

        # percentiles
        relevant_coords = pts[:, coord_idx]
        p10 = np.percentile(relevant_coords, 10)
        p90 = np.percentile(relevant_coords, 90)
        side_range = p90 - p10
        if side_range <= epsilon_flat:
            return "flat"

        perp_coords = pts[:, perp_coord_idx]
        center = pts[len(pts)//2]

        adaptive_pct = max(1, min(10, 100 / len(pts)))
        p10_val = np.percentile(perp_coords, adaptive_pct)
        p10_pt = pts[np.argmin(np.abs(perp_coords-p10_val))]

        if side in ["top", "left"]:
            distance_from_base = centroid[coord_idx] - p10_pt[coord_idx]
            distance_from_mid = centroid[coord_idx] - center[coord_idx]
        else:
            distance_from_base = p10_pt[coord_idx] - centroid[coord_idx]
            distance_from_mid = center[coord_idx] - centroid[coord_idx]

        if distance_from_base > distance_from_mid:
            return "hole"
        else:
            return "knob"

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
