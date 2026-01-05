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

        # download from (huggingface link here)
        self.model = YOLO(model_name).to(device)
        self.device = device

        # does require gated access
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


    def extract_all_edges(self, results, edge_width: int = 20) -> list[dict]:
        """Extract each edge crop for all pieces.
        
        Args:
            results: YOLO segmentation results
        Returns:
            list of dicts of piece idx, side type, and crop
        """
        edge_metadata = []
        
        piece_idx = 0
        for result in results:
            img = result.orig_img
            h, w = img.shape[:2]
            
            for mask in result.masks.data:
                m = mask.detach().cpu().numpy()
                mask_bin = (m>0.5).astype(np.uint8) * 255
                if mask_bin.shape != (h,w):
                    mask_bin = cv2.resize(mask_bin, (w,h), interpolation=cv2.INTER_NEAREST)

                edges = self.extract_sides(mask_bin, img, edge_width=edge_width)
                if edges is None:
                    continue

                # validate then shrink to only whats needed to not add noise to cosim_mat
                for side_name, edge_image in edges.items():
                    if edge_image.size > 0 and edge_image.shape[0] > 0 and edge_image.shape[1] > 0:
                        new_h, new_w = (224,edge_width) if side_name in ["top","bottom"] else (edge_width,224)
                        edge_normalized = cv2.resize(edge_image, (new_h, new_w))
                            
                        edge_metadata.append({
                            "piece_id": piece_idx,
                            "side": side_name,
                            "crop": edge_normalized
                        })
                piece_idx += 1
        return edge_metadata   


    def extract_sides(self, mask, img, edge_width=20) -> dict:
        """Extract 4 edge regions of a puzzle piece (excluding tabs/knobs).
        
        Args:
            mask: binary segmentation mask
            img: original image
            edge_width (int): how many pixels inward from edge to sample
        Returns:
            dict of 4 edge regions with their respective binary mask crop
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # approx a rectangle in the found contour
        # https://www.pythonpool.com/cv2-boundingrect/
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        edges = {}
        edges["top"] = img[y:y+edge_width, x:x+w].copy()
        edges["bottom"] = img[y+h-edge_width:y+h, x:x+w].copy()
        edges["left"] = img[y:y+h, x:x+edge_width].copy()
        edges["right"] = img[y:y+h, x+w-edge_width:x+w].copy()

        # crop it in the real image using a binary mask
        for side_name, edge_crop in edges.items():
            if side_name in ["top", "bottom"]:
                mask_crop = mask[y:y+edge_width if side_name == "top" else y+h-edge_width:y+h, x:x+w]
            else:
                mask_crop = mask[y:y+h, x:x+edge_width if side_name=="left" else x+w-edge_width:x+w]

            mask_crop = cv2.resize(mask_crop, (edge_crop.shape[1], edge_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            edges[side_name] = cv2.bitwise_and(edge_crop, edge_crop, mask=mask_crop)
        return edges 


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


    def crop_segmentations(self, results):
        """UNUSED. TAKE OUT IF EDGE NO WORK
        
        basically crop segs instead of bboxes
        """
        crops = []
        for r in results:
            img = r.orig_img
            h, w = img.shape[:2]
            for mask in r.masks.data:
                m = mask.detach().cpu().numpy()
                mask_bin = (m > 0.5).astype(np.uint8) * 255
                if mask_bin.shape != (h,w):
                    mask_bin = cv2.resize(mask_bin, (w,h), interpolation=cv2.INTER_NEAREST)
                cropped = cv2.bitwise_and(img, img, mask=mask_bin)
                crops.append(cropped)
        return crops


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