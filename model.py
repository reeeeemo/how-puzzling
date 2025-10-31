from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2

class PuzzleImageModel(nn.Module):
    def __init__(self, model_name: str = "best.pt", device: str = "cpu"):
        super().__init__()
        torch.set_float32_matmul_precision("high") # optimize matmuls
        self.model = YOLO(model_name).to(device)
        self.device = device
        
    def forward(self, imgs):
        results = self.model(imgs, verbose=False, imgsz=640)
        return results
    
    def forward_until_layer(self, imgs, layer: int):
        # read all images, resize then normalize into tensors
        all_imgs = []
        for img_path in imgs:
            img = cv2.imread(img_path)[:, :, ::-1] # bgr - rgb
            img = cv2.resize(img, (640,640))
            all_imgs.append((torch.from_numpy(img.copy()).permute(2,0,1).float() / 255.0))
            
        batch = torch.stack(all_imgs).to(self.device)
        
        features = []
        def hook_fn(module, input, output):
            features.append(output)
            
        target_layer = self.model.model.model[layer]
        hook = target_layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = self.model(batch, verbose=False)
        hook.remove()
            
        return features[1]