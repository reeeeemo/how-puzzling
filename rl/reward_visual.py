from model.model import PuzzleImageModel
from dataset.dataset import PuzzleDataset
import argparse
import torch
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
from similarity.similarity import get_compatible_similarities

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

def reward_function(mask_a: np.ndarray,
                    mask_b: np.ndarray,
                    side_a: str,
                    similarity_score: float,
                    dilation_radius: int = 10,
                    expected_contact_ratio: float = 0.07):
    """Given 2 boxes, return the reward for both opposing pieces.

    Reward: whitespace + overlap + top similar
    """
    if mask_a is None or mask_b is None or similarity_score <= 0:
        return -5.0

    height_a, width_a = mask_a.shape
    height_b, width_b = mask_b.shape

    offset_x = {
        "right": width_a,
        "left": -width_b
    }.get(side_a, 0)
    offset_y = {
        "bottom": height_a,
        "top": -height_b
    }.get(side_a, 0)

    padding = 30

    # calc min/max pos
    min_y = min(0, offset_y)
    max_y = max(height_a, height_b + offset_y)
    canvas_h = max_y - min_y + padding * 2

    min_x = min(0, offset_x)
    max_x = max(width_a, width_b + offset_x)
    canvas_w = max_x - min_x + padding * 2

    canvas_a = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_b = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    ya = padding - min_y
    xa = padding - min_x
    canvas_a[ya:ya+height_a, xa:xa+width_a] = mask_a

    yb = ya + offset_y
    xb = xa + offset_x
    canvas_b[yb:yb+height_b, xb:xb+width_b] = mask_b

    cv2.imshow("wahA", canvas_a)
    cv2.imshow("wahB", canvas_b)

    combined = cv2.cvtColor(canvas_a, cv2.COLOR_GRAY2BGR)
    combined[canvas_a > 0] = (0, 0, 255)
    combined[canvas_b > 0] = (255, 0, 0)

    cv2.imshow("combined", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    offset_x = {
        "right": int(width_a * 0.92),
        "left": -int(width_b * 0.92)
    }.get(side_a, 0)
    offset_y = {
        "bottom": int(height_a * 0.92),
        "top": -int(height_b * 0.92)
    }.get(side_a, 0)

    padding = 1  # maybe tunable?
    canvas_h = max(height_a, height_b + abs(offset_y)) + padding * 2
    canvas_w = max(width_a, width_b + abs(offset_x)) + padding * 2

    canvas_a = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_b = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # clip a
    center_y = canvas_h // 2
    center_x = canvas_w // 2

    # ya = padding + max(0, -offset_y)
    # xa = padding + max(0, -offset_x)

    ya = center_y - height_a // 2
    xa = center_x - width_a // 2

    ya_start = max(0, ya)
    ya_end = min(canvas_h, ya + height_a)
    xa_start = max(0, xa)
    xa_end = min(canvas_w, xa + width_a)

    if ya_end > ya_start and xa_end > xa_start:
        mask_a_slice = mask_a[ya_start - ya:ya_start - ya + (ya_end-ya_start),
                              xa_start - xa:xa_start - xa + (xa_end-xa_start)]
        canvas_a[ya_start:ya_end, xa_start:xa_end] = mask_a_slice
    else:
        return -10.0

    # clip b
    yb = ya + offset_y
    xb = xa + offset_x
    yb_start = max(0, yb)
    yb_end = min(canvas_h, yb+height_b)
    xb_start = max(0, xb)
    xb_end = min(canvas_w, xb+width_b)

    if yb_end > yb_start and xb_end > xb_start:
        mask_b_slice = mask_b[yb_start - yb:yb_start - yb + (yb_end-yb_start),
                              xb_start - xb:xb_start - xb + (xb_end-xb_start)]
        canvas_b[yb_start:yb_end, xb_start:xb_end] = mask_b_slice
    else:
        return -10.0

    cv2.imshow("wahA", canvas_a)
    cv2.imshow("wahB", canvas_b)

    combined = cv2.cvtColor(canvas_a, cv2.COLOR_GRAY2BGR)
    combined[canvas_a > 0] = (0, 0, 255)
    combined[canvas_b > 0] = (255, 0, 0)

    cv2.imshow("combined canvas", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return 12*similarity_score


def visualize_reward(model_path: str, images: list):
    """Visualize model results with a reward function piece-wise.

    Args:
        model_path: model to inference with
        images: list of images to inference
    """
    model = PuzzleImageModel(model_name=model_path, device=DEVICE)
    results, similarities, edges = model(images)

    for idx, result in enumerate(results):
        boxes = result.boxes.xyxy
        n_pieces = len(boxes)

        # precompute binary masks
        piece_masks = {}
        for pid in range(n_pieces):
            if pid >= len(results[idx].masks.xy):
                piece_masks[pid] = None
                continue
            poly = results[idx].masks.xy[pid]
            piece_masks[pid] = create_binary_mask(poly,
                                                  boxes[pid],
                                                  images[idx].shape[:2])

        for piece_idx in range(n_pieces):
            piece_edges = [i for i, meta in enumerate(edges)
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

            text_counts = defaultdict(int)
            for edge_idx in piece_edges:
                edge_side = edges[edge_idx]["side"]
                sim_col = similarities[edge_idx, :]

                compatible_sims = get_compatible_similarities(
                    edge_side=edge_side,
                    edge_metadata=edges,
                    cur_piece_idx=piece_idx,
                    similarity_column=sim_col
                )

                for (_, match_meta, sim_score) in compatible_sims[:5]:
                    match_pid = match_meta["piece_id"]
                    match_side = match_meta["side"]

                    if match_pid >= len(boxes):
                        continue

                    # get the matching piece's bbox
                    match_box = map(int, boxes[match_pid])
                    mx1, my1, mx2, my2 = match_box
                    cv2.rectangle(img, (mx1, my1), (mx2, my2), (255, 0, 0), 2)
                    text_y = my1 + 20 + text_counts[match_pid]*20
                    text_counts[match_pid] += 1

                    mask1 = piece_masks.get(piece_idx)
                    mask2 = piece_masks.get(match_pid)
                    reward_score = reward_function(mask1,
                                                   mask2,
                                                   edge_side,
                                                   sim_score)
                    cv2.putText(img,
                                (
                                    f"target:{edge_side}->"
                                    f"this:{match_side}: {reward_score:.3f}"
                                ),
                                (mx1+5, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                                )

            cv2.imshow(f"Edge matches for piece {piece_idx}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def create_binary_mask(poly, box, img_shape: tuple):
    pts = np.asarray(poly, dtype=np.float32)
    if pts.size == 0:
        return

    pts_i = np.rint(pts).astype(np.int32)
    b_mask = np.zeros(img_shape, np.uint8)
    cv2.fillPoly(b_mask, [pts_i], 255)

    x1, y1, x2, y2 = map(int, box)
    b_crop = b_mask[y1:y2, x1:x2]

    return b_crop


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
