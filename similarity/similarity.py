from pathlib import Path
import torch
import numpy as np
import cv2
from model.model import PuzzleImageModel
from dataset.dataset import PuzzleDataset
import argparse
from collections import defaultdict
from utils.polygons import create_binary_mask, get_polygon_sides

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Computes edge-to-edge similarity amongst all puzzle pieces
# using a trained image segmentation model, and dataset split.
# Outputs top 5 per puzzle piece

# Use (root): python -m similarity.similarity
#   --dataset <dataset_path>
#   --model <model_path>
#   --split <split>
# Example:
# python -m similarity.similarity
#   --dataset dataset/data/jigsaw_puzzle
#   --model model/puzzle-segment-model/best.pt
#   --split test


def compute_similarites(
        model: PuzzleImageModel,
        dataset_path: Path,
        split: str):
    """Get the similarity matrix of the requested split.

    Args:
        model_path: path to pretrained segmentation model
        dataset_path: path to dataset of images in YOLO format
         split: split to take images from
    Returns:
        tuple consisting of:
            YOLO-style results from seg model,
            cosine similarity matrix,
            all images that were segmented,
            dict of all edge data.
    """

    dataset = PuzzleDataset(root_dir=dataset_path,
                            splits=[split],
                            extension="jpg")

    all_test_images = [
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for img, _ in dataset
    ]
    results, similarities, edge_metadata = model(all_test_images)
    return results, similarities, all_test_images, edge_metadata


def visualize_edge_crop(crop,
                        side_name: str,
                        piece_id: int):
    """Create a visualization of an edge crop with label.

    Args:
        crop: edge crop to visualize
        side_name: name to accompany visualization
        piece_id: ID to accompany visualization
    Returns:
        resized crop with added metadata text.
    """
    if side_name in ["top", "bottom"]:
        vis_crop = cv2.resize(crop, (224, 60))
    else:
        vis_crop = cv2.resize(crop, (60, 224))

    vis_crop = cv2.copyMakeBorder(vis_crop, 20, 5, 5, 5,
                                  cv2.BORDER_CONSTANT,
                                  value=(50, 50, 50))
    cv2.putText(vis_crop,
                f"P{piece_id}:{side_name}",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255), 1)

    return vis_crop


def get_compatible_similarities(edge_side: str,
                                edge_metadata: dict,
                                cur_piece_idx: int,
                                poly_sides: dict,
                                piece_masks: dict,
                                cur_piece_type: str,
                                cur_piece_sides: dict,
                                model: PuzzleImageModel,
                                similarity_column):
    """Compute opposing edge similarities if valid, rank then return.

    Valid puzzle piece edges are of different types, and do not break
    any jigsaw puzzle piece rules (no out of bounds pieces).

    Args:
        edge_side: current edge side to compare against
        edge_metadata: information about every edge
        cur_piece_idx: current puzzle piece ID
        poly_sides: all dicts of sides, list of pts of puzzle pieces
        piece_masks: dict of binary masks for every puzzle piece
        cur_piece_type: current type of the puzzle piece (side, corner, etc.)
        cur_piece_sides: dict of types for each side of a piece (knob, hole)
        model: Model to run inference on
        similarity_column: cosine sim mat for all edges
    Returns:
        list of compatible similarities, ranked.
    """
    opposite = {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left"
    }[edge_side]

    compatible_edges = []

    for i, meta in enumerate(edge_metadata):
        if meta["side"] == opposite and meta["piece_id"] != cur_piece_idx:
            match_pid = meta["piece_id"]
            match_side = meta["side"]
            match_mask = piece_masks.get(match_pid)
            centroid = model.get_centroid(match_mask, binary_mask=True)

            match_type, match_sides = model.classify_piece(
                poly_sides.get(match_pid),
                centroid
            )

            # side-to-side can only be same-border
            if (
                match_type.startswith("side_") and
                cur_piece_type.startswith("side_") and
                match_type != cur_piece_type
            ):
                continue

            # no knob -> knob or hole -> hole
            if (match_sides.get(match_side) ==
               cur_piece_sides.get(edge_side)):
                continue
            compatible_edges.append((i, meta))

    if not compatible_edges:
        return None

    compatible_sims = [
        (i, meta, similarity_column[i].item())
        for i, meta in compatible_edges
    ]
    compatible_sims.sort(key=lambda x: x[2], reverse=True)
    return compatible_sims


def plot_n_similar_edges(sim_mat,
                         n: int,
                         bboxes: list,
                         img,
                         edge_side: str,
                         text_counts: dict):
    """Plot on an img the top n edge similarities.

    Args:
        sim_mat: cosine sim mat for all edges
        n: N sims to plot
        bboxes: list of bboxes
        img: image to plot on
        edge_side: current edge side
        text_counts: dict to align y axis text
    """

    # top 5 matches
    for (rank,
         (_, match_meta, sim_score)
         ) in enumerate(sim_mat[:n]):
        match_pid = match_meta["piece_id"]
        match_side = match_meta["side"]

        mx1, my1, mx2, my2 = map(int, bboxes[match_pid])

        if rank == 0:
            color = (0, 255, 0)
        elif rank == 1:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        cv2.rectangle(img, (mx1, my1), (mx2, my2), color, 2)
        text_y = my1 + 20 + text_counts[match_pid]*25
        text_counts[match_pid] += 1

        cv2.putText(img,
                    (
                        f"target:{edge_side}->"
                        f"this:{match_side}: {sim_score:.3f}"
                    ),
                    (mx1+5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )


def visualize_similarities(results: list,
                           model: PuzzleImageModel,
                           sims,
                           images: list,
                           edge_metadata: dict):
    """Visualize model results piece-wise.

    Args:
        results: YOLO-style results list
        model: model to run inference on
        sims: cosine sim mat of edge similarities
        images: list of images inferenced
        edge_metadata: info about every edge
    """
    for idx, result in enumerate(results):
        boxes = result.boxes.xyxy

        # precompute binary mask + polygon sides
        poly_sides = {}
        piece_masks = {}
        for pid in range(len(boxes)):
            if pid >= len(results[idx].masks.xy):
                piece_masks[pid] = None
                poly_sides[pid] = None
                continue
            poly = result.masks.xy[pid]
            piece_masks[pid] = create_binary_mask(
                poly=poly, box=boxes[pid], img_shape=images[idx].shape[:2]
            )
            poly_sides[pid] = get_polygon_sides(
                poly=poly, bbox=boxes[pid], model=model
            )

        for piece_idx in range(len(boxes)):
            piece_edges = [i for i, meta in enumerate(edge_metadata)
                           if meta["piece_id"] == piece_idx]
            if not piece_edges:
                continue

            img = images[idx].copy()
            x1, y1, x2, y2 = map(int, boxes[piece_idx])

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.putText(img,
                        f"Piece {piece_idx}",
                        (x1+5, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255), 2)

            # get current piece info once
            cur_piece_centroid = model.get_centroid(
                piece_masks[piece_idx],
                binary_mask=True
            )
            cur_piece_type, cur_piece_sides = model.classify_piece(
                edge_metadata=poly_sides.get(piece_idx),
                centroid=cur_piece_centroid
            )

            text_counts = defaultdict(int)
            edge_crops_display = []
            for edge_idx in piece_edges:
                edge_side = edge_metadata[edge_idx]["side"]
                edge_crop = edge_metadata[edge_idx]["crop"]
                sim_col = sims[edge_idx, :]

                compatible_sims = get_compatible_similarities(
                    edge_side=edge_side,
                    poly_sides=poly_sides,
                    piece_masks=piece_masks,
                    cur_piece_type=cur_piece_type,
                    cur_piece_sides=cur_piece_sides,
                    model=model,
                    edge_metadata=edge_metadata,
                    cur_piece_idx=piece_idx,
                    similarity_column=sim_col
                )

                current_edge_vis = visualize_edge_crop(edge_crop,
                                                       edge_side,
                                                       piece_idx)
                edge_crops_display.append(current_edge_vis)

                plot_n_similar_edges(
                    sim_mat=compatible_sims,
                    n=5,
                    bboxes=boxes,
                    img=img,
                    edge_side=edge_side,
                    text_counts=text_counts
                )

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


def main():
    parser = argparse.ArgumentParser(
        prog="Edge-To-Edge Similarity of a Dataset Split"
    )
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="filepath of dataset")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="filepath of model")
    parser.add_argument("--split",
                        type=str,
                        required=True,
                        help="split to inference")
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parent.parent
    model_path = project_path / args.model
    dataset_path = project_path / args.dataset

    model = PuzzleImageModel(model_name=str(model_path), device=DEVICE)
    (
        results,
        sims,
        images,
        edge_metadata
    ) = compute_similarites(model, dataset_path, args.split)
    visualize_similarities(
        results=results,
        model=model,
        sims=sims,
        images=images,
        edge_metadata=edge_metadata
    )


if __name__ == "__main__":
    main()
