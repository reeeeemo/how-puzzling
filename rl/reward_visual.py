from model.model import PuzzleImageModel
from dataset.dataset import PuzzleDataset
import argparse
import torch
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
from similarity.similarity import get_compatible_similarities
import heapq
from utils.polygons import get_polygon_sides, create_binary_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Visualizes an example of the reward function to be used
# during the agent's learning process

# Use (root): python -m rl.reward_visual
#   --dataset <dataset_path>
#   --model <model_path>
#   --split <split>
# Example:
# python -m rl.reward_visual
#   --dataset dataset/data/jigsaw_puzzle
#   --model model/puzzle-segment-model/best.pt
#   --split test

def pad_offset_to_size(img: np.ndarray, target_size: tuple, offset: tuple):
    """Pads an image to the target size, centers based on an offset.

    Args:
        img: image to pad.
        target_size (height, width): how much padding in height, width format.
        offset (y_offset, x_offset): position of bottom left corner
    Returns:
        padded / centered image.
    """
    target_h, target_w = target_size
    h, w = img.shape[:2]
    y_offset, x_offset = offset

    pad_top = max(0, y_offset)
    pad_bottom = max(0, (target_h - h - pad_top))
    pad_left = max(0, x_offset)
    pad_right = max(0, (target_w - w - pad_left))

    padded = cv2.copyMakeBorder(
        img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return padded


def reward_function(mask_a: np.ndarray,
                    mask_b: np.ndarray,
                    pts_a: dict,
                    pts_b: dict,
                    side_a: str,
                    side_b: str,
                    model: PuzzleImageModel,
                    similarity_score: float):
    """Given 2 boxes, return the reward for both opposing pieces.

    Args:
        mask_a: current piece mask
        mask_b: piece mask to match
        pts_a: dict of piece side, ordered pts
        pts_b: dict of match side, ordered pts
        side_a: current piece side
        side_b: matching piece side
        model: model to use for inference
        similarity_score: sim score between 2 edges
    Returns:
        Reward: whitespace + overlap + top similar
    """
    if mask_a is None or mask_b is None or similarity_score <= 0:
        return -5.0

    h_a, w_a = mask_a.shape[:2]
    h_b, w_b = mask_b.shape[:2]

    # polygon centroid
    centroid_a = model.get_centroid(mask_a, binary_mask=True)
    centroid_b = model.get_centroid(mask_b, binary_mask=True)
    if centroid_a is None or centroid_b is None:
        return -5.0

    # current side polygon to check
    side_a_pts = pts_a.get(side_a, np.array([]))
    if len(side_a_pts) == 0:
        return -5.0
    side_b_pts = pts_b.get(side_b, np.array([]))
    if len(side_b_pts) == 0:
        return -5.0

    cx_a, cy_a = int(centroid_a[0]), int(centroid_a[1])
    cx_b, cy_b = int(centroid_b[0]), int(centroid_b[1])
    padding = 50
    overlap_percentage_x = 0.27
    overlap_percentage_y = 0.3
    k = 0.4

    # compute max width/height,
    # offset by the mean of 15% of the middle points
    # and offset by any padding when making same-size images
    # finally, overlap both pieces by 30% of the distance between them
    if side_a in ["right", "left"]:
        n_a = len(side_a_pts[:, 1])
        n_b = len(side_b_pts[:, 1])
        mid_y_a = side_a_pts[:, 1][
            int(n_a // 2 - (n_a * k)):int(n_a // 2 + (n_a * k))
        ].mean()
        mid_y_b = side_b_pts[:, 1][
            int(n_b // 2 - (n_b * k)):int(n_b // 2 + (n_b * k))
        ].mean()
        dist_a = abs(cx_a - side_a_pts[:, 0].mean())
        dist_b = abs(cx_b - side_b_pts[:, 0].mean())
        overlap_distance_x = int((dist_a + dist_b) * overlap_percentage_x)
        # overlap_distance_x = max(30, min(120, overlap_distance_x))

        canvas_w = w_a + w_b + padding * 2
        canvas_h = max(h_a, h_b) + padding * 2

        offset_a_y = int(padding + (canvas_h - padding * 2) // 2 - cy_a)
        if side_a == "right":
            offset_a_x = padding
            offset_b_x = padding + w_a - overlap_distance_x
        else:
            offset_a_x = padding + w_b - overlap_distance_x
            offset_b_x = padding

        offset_b_y = int(offset_a_y + (mid_y_a - mid_y_b))
    else:
        n_a = len(side_a_pts[:, 0])
        n_b = len(side_b_pts[:, 0])

        mid_x_a = side_a_pts[:, 0][
            int(n_a // 2 - (n_a * k)):int(n_a // 2 + (n_a * k))
        ].mean()
        mid_x_b = side_b_pts[:, 0][
            int(n_b // 2 - (n_b * k)):int(n_b // 2 + (n_b * k))
        ].mean()
        dist_a = abs(cy_a - side_a_pts[:, 1].mean())
        dist_b = abs(cy_b - side_b_pts[:, 1].mean())
        overlap_distance_y = int((dist_a + dist_b) * overlap_percentage_y)
        # overlap_distance_y = max(30, min(120, overlap_distance_y))
        canvas_w = max(w_a, w_b) + padding * 2
        canvas_h = h_a + h_b + padding * 2

        offset_a_x = int(padding + (canvas_w - padding * 2) // 2 - cx_a)

        if side_a == "bottom":
            offset_a_y = padding
            offset_b_y = padding + h_a - overlap_distance_y
        else:
            offset_a_y = padding + h_b - overlap_distance_y
            offset_b_y = padding

        offset_b_x = int(offset_a_x + (mid_x_a - mid_x_b))
        offset_b_x = max(0, min(offset_b_x, canvas_w - w_b))

    # pad actual masks to the correct size and move to offset
    padded_a = pad_offset_to_size(mask_a,
                                  (canvas_h, canvas_w),
                                  (offset_a_y, offset_a_x))
    padded_b = pad_offset_to_size(mask_b,
                                  (canvas_h, canvas_w),
                                  (offset_b_y, offset_b_x))

    side_xs = side_a_pts[:, 0] + offset_a_x
    side_ys = side_a_pts[:, 1] + offset_a_y
    edge_buffer = 10

    edge_min_x = max(0, int(side_xs.min()) - edge_buffer)
    edge_max_x = min(canvas_w-1, int(side_xs.max()) + edge_buffer)
    edge_min_y = max(0, int(side_ys.min()) - edge_buffer)
    edge_max_y = min(canvas_h-1, int(side_ys.max()) + edge_buffer)

    # compute overlap of both edges given their bbox
    region_a = padded_a[edge_min_y:edge_max_y, edge_min_x:edge_max_x]
    region_b = padded_b[edge_min_y:edge_max_y, edge_min_x:edge_max_x]

    # https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    # https://docs.opencv.org/3.4/d5/d45/tutorial_py_contours_more_functions.html
    edge_a = cv2.Canny(region_a, 100, 200)
    edge_b = cv2.Canny(region_b, 100, 200)

    edge_distance = cv2.matchShapes(
        edge_a,
        edge_b,
        cv2.CONTOURS_MATCH_I1,
        0
    )

    # visualize the masks
    combined = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    combined[padded_a > 0] = (0, 0, 128)
    combined[padded_b > 0] = (128, 0, 0)
    combined[(padded_a > 0) & (padded_b > 0)] = (255, 255, 255)

    cv2.rectangle(combined, (edge_min_x, edge_min_y),
                  (edge_max_x, edge_max_y), (255, 255, 255), 2)

    for pt in side_a_pts:
        cv2.circle(combined,
                   (int(pt[0])+offset_a_x, int(pt[1])+offset_a_y),
                   2,
                   (0, 0, 255),
                   2)
    for pt in side_b_pts:
        cv2.circle(combined,
                   (int(pt[0])+offset_b_x, int(pt[1])+offset_b_y),
                   2,
                   (255, 0, 0),
                   2)

    # cv2.imshow(f"pieces aligned on {side_a} axis", combined)
    # cv2.waitKey(0)

    return (
        (similarity_score * 1000) -
        (1000 * edge_distance)
    )


def visualize_reward(model_path: str, images: list):
    """Visualize model results with a reward function piece-wise.

    Args:
        model_path: model to inference with
        images: list of images to inference
    """
    model = PuzzleImageModel(model_name=model_path, device=DEVICE)
    results, similarities, edge_metadata = model(images)

    for idx, result in enumerate(results):
        boxes = result.boxes.xyxy
        n_pieces = len(boxes)

        # precompute binary masks, polygon sides, and piece types
        poly_sides = {}
        piece_masks = {}
        piece_classifications = {}
        for pid in range(n_pieces):
            if pid >= len(results[idx].masks.xy):
                piece_masks[pid] = None
                poly_sides[pid] = None
                continue
            poly = results[idx].masks.xy[pid]
            piece_masks[pid] = create_binary_mask(
                poly,
                boxes[pid],
                images[idx].shape[:2]
            )
            poly_sides[pid] = get_polygon_sides(
                poly=poly,
                bbox=boxes[pid],
                model=model
            )
            piece_edges = [
                meta for meta in edge_metadata
                if meta["piece_id"] == pid and meta["image_id"] == idx
            ]
            if piece_edges:
                piece_type, piece_sides = model.classify_piece(
                    edge_metadata=piece_edges
                )
                piece_classifications[pid] = (piece_type, piece_sides)

        for piece_idx in range(n_pieces):
            piece_edges = [
                i for i, meta in enumerate(edge_metadata)
                if meta["piece_id"] == piece_idx
                and meta["image_id"] == idx
            ]

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

            cur_piece = poly_sides.get(piece_idx)
            cur_piece_type, cur_piece_sides = piece_classifications[piece_idx]

            text_counts = defaultdict(int)
            all_matches = []
            for edge_idx in piece_edges:
                edge_side = edge_metadata[edge_idx]["side"]
                sim_col = similarities[edge_idx, :]

                compatible_sims = get_compatible_similarities(
                    edge_side=edge_side,
                    edge_metadata=edge_metadata,
                    cur_piece_idx=piece_idx,
                    cur_image_idx=idx,
                    piece_classifications=piece_classifications,
                    cur_piece_type=cur_piece_type,
                    cur_piece_sides=cur_piece_sides,
                    similarity_column=sim_col
                )

                if not compatible_sims:
                    continue

                top_idxs = get_top_n_compatible_sims(
                    compatible_sims=compatible_sims,
                    polys=poly_sides,
                    piece_masks=piece_masks,
                    cur_piece=cur_piece,
                    cur_piece_mask=piece_masks.get(piece_idx),
                    cur_edge=edge_side,
                    model=model,
                    n=5
                )

                for reward, compat_idx in top_idxs:
                    match_pid = compatible_sims[compat_idx][1]["piece_id"]
                    match_side = compatible_sims[compat_idx][1]["side"]
                    all_matches.append((
                        reward, edge_idx,
                        compat_idx, match_pid,
                        match_side, edge_side
                    ))

            all_matches.sort(reverse=True, key=lambda x: x[0])
            assigned_edges = set()
            assigned_pieces = set()
            text_counts = defaultdict(int)

            for (
                reward, edge_idx,
                compat_idx, match_pid,
                match_side, edge_side
            ) in all_matches:
                if edge_idx in assigned_edges or match_pid in assigned_pieces:
                    continue

                assigned_edges.add(edge_idx)
                assigned_pieces.add(match_pid)

                # place highest rated pid
                mx1, my1, mx2, my2 = map(int, boxes[match_pid])
                cv2.rectangle(img, (mx1, my1), (mx2, my2), (255, 0, 0), 2)
                text_y = my1 + 20 + text_counts[match_pid]*20
                text_counts[match_pid] += 1
                cv2.putText(img,
                            (
                                f"target:{edge_side}->"
                                f"this:{match_side}: {reward:.3f}"
                            ),
                            (mx1+5, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
                            )

            cv2.imshow(f"Edge matches for piece {piece_idx}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def get_top_n_compatible_sims(
        compatible_sims: list[tuple],
        polys: dict,
        piece_masks: dict,
        cur_piece: np.ndarray,
        cur_piece_mask: np.ndarray,
        cur_edge: str,
        model: PuzzleImageModel,
        n: int
     ) -> list[tuple]:
    top_n_piece_idxs = []

    for compat_idx, (_,
                     match_meta,
                     sim_score) in enumerate(compatible_sims[:5]):
        match_pid = match_meta["piece_id"]
        match_side = match_meta["side"]
        match_mask = piece_masks.get(match_pid)

        match_piece = polys.get(match_pid)

        reward_score = reward_function(
            mask_a=cur_piece_mask,  # piece_masks.get(piece_idx),
            mask_b=match_mask,
            pts_a=cur_piece,
            pts_b=match_piece,
            side_a=cur_edge,
            side_b=match_side,
            model=model,
            similarity_score=sim_score
        )

        if len(top_n_piece_idxs) < n:
            heapq.heappush(top_n_piece_idxs, (reward_score, compat_idx))
        elif reward_score > top_n_piece_idxs[0][0]:
            heapq.heapreplace(top_n_piece_idxs, (reward_score, compat_idx))
    return sorted(top_n_piece_idxs, reverse=True)


def main():
    parser = argparse.ArgumentParser(
        prog="Visualization of RL Reward Function"
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

    dataset = PuzzleDataset(root_dir=dataset_path,
                            splits=[args.split],
                            extension="jpg")

    all_split_images = [
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for img, _ in dataset
    ]

    visualize_reward(model_path, all_split_images)


if __name__ == "__main__":
    main()
