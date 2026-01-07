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
        self.similarity_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.similarity_model = AutoModel.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )


    def forward(self, imgs) -> tuple[torch.Tensor, dict[int, list], list[dict]]:
        """Segment and compute edge-to-edge similarity on all puzzle images.
        
        Args:
            imgs: list of numpy arrays
        Returns:
            Tuple of cosine similarities of each detected object,
            Xyxy coords of boxes cropped relative to segmented mask,
            List of dicts tracking piece, side, and crop for every image.
        """
        results = self.model(imgs, verbose=False)
        edge_metadata = self.extract_all_edges(results, edge_width=60)
        boxes_per_image = self.crop_images(results, imgs)
        similarities = self.compute_similarities([data.get("crop") for data in edge_metadata])
        return similarities, boxes_per_image, edge_metadata


    def extract_all_edges(self, results, edge_width: int = 20) -> dict:
        """Extract each edge crop for all pieces.
        
        Args:
            results: YOLO segmentation results
        Returns:
            list of dicts of piece idx, side type, and crop
        """
        edge_metadata = []

        sides = {
            "bottom": (0,1),
            "top": (0,-1),
            "left": (-1,0),
            "right": (1,0)
        }
        
        for result in results:  # all image segmentation results
            img = result.orig_img
            h, w = img.shape[:2]

            piece_idx = 0
            for poly in getattr(result.masks, "xy", []):  # for each polygon inside image
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
                mu = cv2.moments(pts_i.reshape(-1,1,2))
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
                        ((name, float(np.dot(radial, np.asarray(side, dtype=np.float64)))) for name, side in sides.items()),
                        key=lambda t: t[1]
                    )
                    all_pts[best_side].append(cur_pt)
            
                for side_name, pts_side in all_pts.items():
                    if not pts_side:
                        continue
                    arr = np.asarray(pts_side)
                    x_min, x_max = arr[:, 0].min(), arr[:, 0].max()
                    y_min, y_max = arr[:, 1].min(), arr[:, 1].max()

                    if side_name in ("top", "bottom"):
                        center_y = int(y_min if side_name == "top" else y_max)
                        y1 = max(0, center_y - edge_width)
                        y2 = min(h, center_y + edge_width)
                        x1 = int(max(0, np.floor(x_min)))
                        x2 = int(min(w, np.ceil(x_max)))
                    else:
                        center_x = int(x_min if side_name == "left" else x_max)
                        x1 = max(0, center_x - edge_width)
                        x2 = min(w, center_x + edge_width)
                        y1 = int(max(0, np.floor(y_min)))
                        y2 = int(min(h, np.ceil(y_max)))
                    
                    if y2 <= y1 or x2 <= x1:
                        continue

                    crop = img[y1:y2, x1:x2].copy()
                    mask_crop = mask_piece[y1:y2, x1:x2]
                    if mask_crop.size == 0:
                        continue

                    mask_bin = (mask_crop > 0).astype(np.uint8) * 255
                    crop = cv2.bitwise_and(crop, crop, mask=mask_bin)

                    # normalize crop shape for embedding model
                    try:
                        if side_name in ("top", "bottom"):
                            crop_resized = cv2.resize(crop, (224, edge_width))
                        else:
                            crop_resized = cv2.resize(crop, (edge_width, 224))
                    except Exception:
                        crop_resized = crop
                    
                    edge_metadata.append({
                        "piece_id": piece_idx,
                        "side": side_name,
                        "crop": crop_resized,
                        "coords": (x1, y1, x2, y2)
                    })
                piece_idx += 1

        return edge_metadata


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
            p1 = pts[(i+1)%len(pts)]  # nxt point or 0

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
            inputs = self.similarity_processor(images=imgs, return_tensors="pt").to(self.device)
            outputs = self.similarity_model(**inputs)

        return outputs


    def compute_similarities(self, imgs):
        """
            Compute a similarity matrix of multiple images using cosine similarity
            Args:
                imgs: list of numpy arrays
            Returns:
                Tensor of img-to-img cosine similarities
        """
        outputs = self.compute_embeddings(imgs)
        image_feats = outputs.last_hidden_state[:, 0, :]  # dinov3 has overall cls embedding
        image_feats = nn.functional.normalize(image_feats, dim=1)
        
        # cos sim is (a \dot b) / ||a|| x ||b||
        # we normalize so magnitude is 1 which reduces to a \dot b
        sim_mat = torch.mm(image_feats, image_feats.T)
        sim_mat = torch.clamp(sim_mat, 0.0, 1.0)
        sim_mat.fill_diagonal_(1.0)
        return sim_mat


    def crop_images(self, results, imgs):
        """
            Crop the images per segmentation
            Args:
                results: segmentation masks from model
                imgs: list of numpy arrays
            Returns:
                tuple: tensor of batched cropped,
                xyxy coords of cropped box
        """
        #all_crops = []
        boxes_per_image = {i: [] for i in range(len(imgs))}
        
        for idx, (result, img) in enumerate(zip(results, imgs)):
            boxes = result.boxes.xyxy
            h, w = img.shape[:2]
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(w-1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h-1,y1))
                y2 = max(0, min(h,y2))
                
                if x2 <= x1 or y2 <= y1:
                    continue # boxes out of bounds
                
                #crop = img[y1:y2, x1:x2]
                #crop_t = torch.from_numpy(cv2.resize(crop, (640,640))).permute(2,0,1).float() / 255.0
                #all_crops.append(crop_t)
                boxes_per_image[idx].append((x1, y1, x2, y2))
                
        #if not all_crops:
        #    return boxes_per_image # torch.empty(0, 3, 640, 640), 
                        
        return boxes_per_image # torch.stack(all_crops), 