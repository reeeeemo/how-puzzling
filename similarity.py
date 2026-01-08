from pathlib import Path
import torch
import numpy as np
import cv2
from model.model import PuzzleImageModel
from dataset.dataset import PuzzleDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_similarites(model_path: Path, dataset_path: Path, split: str):
    """Get the similarity matrix of the requested split.
    
    Args:
        model_path: path to pretrained segmentation model
        dataset_path: path to dataset of images in YOLO format
         split: split to take images from
    Returns:
        tuple of (cosine similarity matrix,
        all images from split that was segmented,
        xyxy coords of boxes cropped relative to segmented masks
        list of dicts tracking which piece/side for each edge crop).
    """

    model = PuzzleImageModel(model_name=str(model_path), device=DEVICE)
    dataset = PuzzleDataset(root_dir=dataset_path, splits=[split], extension="jpg")

    all_test_images = [
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for img, _ in dataset
    ]
    similarities, boxes_per_image, edge_metadata = model(all_test_images)
    return similarities, all_test_images, boxes_per_image, edge_metadata


def visualize_edge_crop(crop, side_name, piece_id):
    """Create a visualization of an edge crop with label."""
    if side_name in ["top", "bottom"]:
        vis_crop = cv2.resize(crop, (224, 60))
    else:
        vis_crop = cv2.resize(crop, (60, 224))

    vis_crop = cv2.copyMakeBorder(vis_crop, 20, 5, 5, 5, cv2.BORDER_CONSTANT, value=(50,50,50))
    cv2.putText(vis_crop, f"P{piece_id}:{side_name}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    return vis_crop


def main():
    project_path = Path(__file__).resolve().parent
    model_path = project_path / "model" / "puzzle-segment-model" / "best.pt"
    dataset_path = project_path / "dataset" / "data" / "jigsaw_puzzle"


    sims, images, boxes_per_image, edge_metadata = compute_similarites(model_path, dataset_path, "test")

    n_pieces = max(meta["piece_id"] for meta in edge_metadata) + 1

    for idx, boxes in boxes_per_image.items():
        for piece_idx in range(n_pieces):
            piece_edges = [i for i, meta in enumerate(edge_metadata)
                           if meta["piece_id"] == piece_idx]
            if not piece_edges:
                continue

            img = images[idx].copy()
            x1, y1, x2, y2 = boxes[piece_idx]

            cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 3)
            cv2.putText(img, f"Piece {piece_idx}", (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            
            text_counts = {}
            edge_crops_display = []
            for edge_idx in piece_edges:
                edge_side = edge_metadata[edge_idx]["side"]
                edge_crop = edge_metadata[edge_idx]["crop"]
                sim_col = sims[edge_idx, :]
                
                if edge_side in ["top", "bottom"]:
                    opposite = "bottom" if edge_side == "top" else "top"
                else:
                    opposite = "right" if edge_side == "left" else "left"

                compatible_edges = [
                    (i, meta) for i, meta in enumerate(edge_metadata)
                    if meta["side"] == opposite and meta["piece_id"] != piece_idx
                ]
                
                if not compatible_edges:
                    continue
                
                compatible_sims = [
                    (i, meta, sim_col[i].item())
                    for i, meta in compatible_edges
                ]
                compatible_sims.sort(key=lambda x: x[2], reverse=True)
                
                current_edge_vis = visualize_edge_crop(edge_crop, edge_side, piece_idx)
                edge_crops_display.append(current_edge_vis)
                
                # top 5 matches
                for rank, (_, match_meta, sim_score) in enumerate(compatible_sims[:5]):
                    match_piece_id = match_meta["piece_id"]
                    match_side = match_meta["side"]
                    
                    if match_piece_id >= len(boxes):
                        continue
                    match_box = boxes[match_piece_id]
                    mx1, my1, mx2, my2 = match_box
                    
                    if match_piece_id not in text_counts:
                        text_counts[match_piece_id] = 0

                    if rank == 0:
                        color = (0,255,0)
                    elif rank == 1:
                        color = (0,255,255)
                    else:
                        color = (0, 165, 255)
                        
                    cv2.rectangle(img, (mx1,my1), (mx2,my2), color, 2)
                    text_y = my1 + 20 + text_counts[match_piece_id]*25
                    text_counts[match_piece_id] += 1
                    
                    cv2.putText(img, f"{edge_side}->{match_side}: {sim_score:.3f}", (mx1+5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if edge_crops_display:
                max_height = img.shape[0]
                crop_panel = np.zeros((max_height, 300, 3), dtype=np.uint8)
                y_offset = 10
                
                for crop_vis in edge_crops_display:
                    h, w = crop_vis.shape[:2]
                    if y_offset+h < max_height and w <= 290:
                        crop_panel[y_offset:y_offset+h, 10:10+w] = crop_vis
                        y_offset += h + 10
                        
                combined = np.hstack([img, crop_panel])  
                cv2.imshow(f"Edge matches for piece {piece_idx}", combined)
            else:
                cv2.imshow(f"Edge matches for piece {piece_idx}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return
    
if __name__ == "__main__":
    main()