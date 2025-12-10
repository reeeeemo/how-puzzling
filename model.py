from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2
import numpy as np

###
# gotta redo this
###

class PuzzleImageModel(nn.Module):
    def __init__(self, model_name: str = "best.pt", device: str = "cpu"):
        super().__init__()
        torch.set_float32_matmul_precision("high") # optimize matmuls
        self.model = YOLO(model_name).to(device)
        self.device = device
        
    def forward(self, imgs):
        """
            Forward pass of the model
            Args:
                img_paths: list of numpy arrays
            Returns:
                Cosine similarities of each detected object relative to the current image
        """
        results = self.model(imgs, verbose=False)
        cropped = self.crop_images(results, imgs)

        # get features from pre-detect layer
        cropped_images = list(crop for (_,_, crop) in cropped)
        feats = self.forward_until_layer(cropped_images, -2)
        
        # global avg pool + cosine similarity
        global_avg_pool = torch.mean(feats, dim=[2,3]) # mean over width/height
        
        pooled = torch.nn.functional.normalize(global_avg_pool, dim=1)
        sims = pooled @ pooled.T 
                
        return sims.cpu().numpy()
    
    def crop_images(self, results, imgs):
        """
            Crop the images per box/class
            Args:
                results: segmentation masks from model
                imgs: list of numpy arrays
            Returns:
                List of tuples (img_idx, cls, crop)
        """
        cropped = []
        for i, (result, img) in enumerate(zip(results, imgs)):
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            
            h, w = img.shape[:2]
            
            # iterate over all boxes and classes
            for (box, cls_val) in zip(boxes, classes):
                # no boxes going out of screen
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(w-1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h-1,y1))
                y2 = max(0, min(h,y2))
                
                if x2 <= x1 or y2 <= y1:
                    continue # :<
                
                crop = img[y1:y2, x1:x2]
                
                crop_t = torch.from_numpy(cv2.resize(crop, (640,640))).permute(2,0,1).float() / 255.0
                cropped.append((i, cls_val.item(), crop_t))
                
        return cropped
                
    def forward_until_layer(self, imgs, layer: int):
        # resize all images and normalize into tensors
        batch = torch.stack(imgs).to(self.device)
        
        features = []
        def hook_fn(module, input, output):
            features.append(output)
            
        target_layer = self.model.model.model[layer]
        hook = target_layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = self.model(batch, verbose=False)
        hook.remove()
            
        return features[0]