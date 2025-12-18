from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel


class PuzzleImageModel(nn.Module):
    """
        Segments all individual jigsaw puzzle pieces in an image, then computes
        similarity between them
        Attributes:
        model: pretrained YOLO model to segment images with
        device: whether device uses cuda or cpu
        similarity_processor: DINOv3 image encoder's processor
        similarity_model: DINOv3 image encoder
    """
    def __init__(self, model_name: str = "best.pt", device: str = "cpu"):
        super().__init__()
        torch.set_float32_matmul_precision("high") # optimize matmuls
        self.model = YOLO(model_name).to(device)
        self.device = device
        
        self.similarity_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.similarity_model = AutoModel.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        
    def forward(self, imgs):
        """
            Forward pass of the model
            Args:
                imgs: list of numpy arrays
            Returns:
                tuple of cosine similarities of each detected object,
                xyxy coords of boxes cropped relative to segmented mask
        """
        results = self.model(imgs, verbose=False)
        cropped, boxes_per_image = self.crop_images(results, imgs)
        
        similarities = self.compute_similarities(cropped)
        return similarities, boxes_per_image

    def compute_embeddings(self, imgs):
        """
            Compute embeddings of an image using the encoder
            Args:
                imgs: list of numpy arrays
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
        """
        outputs = self.compute_embeddings(imgs)
        image_feats = outputs.last_hidden_state[:, 0, :] # extract cls token embedding
        image_feats = nn.functional.normalize(image_feats, dim=1)
        
        # cos sim is (a \dot b) / ||a|| x ||b||, but we normalize so magnitude is 1 which reduces to
        # a \dot b
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
        all_crops = []
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
                
                crop = img[y1:y2, x1:x2]
                crop_t = torch.from_numpy(cv2.resize(crop, (640,640))).permute(2,0,1).float() / 255.0
                all_crops.append(crop_t)
                boxes_per_image[idx].append((x1, y1, x2, y2))
                
        if not all_crops:
            return torch.empty(0, 3, 640, 640), boxes_per_image
                        
        return torch.stack(all_crops), boxes_per_image