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

def get_centroid(mask: np.ndarray) -> tuple[int, int]:
    """Calculate centroid of a binary mask."""
    mu = cv2.moments(mask)
    if mu.get('m00', 0) == 0:
        return None
    return (
        int(mu["m01"] / mu["m00"]),  # cy
        int(mu["m10"] / mu["m00"])  # cx
    )


def pad_center_to_size(img: np.ndarray, target_size: tuple, offset: tuple):
    """gaga wuwu wawa.

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

    pad_top = y_offset
    pad_bottom = target_h - h - pad_top
    pad_left = x_offset
    pad_right = target_w - w - pad_left

    padded = cv2.copyMakeBorder(
        img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return padded


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

    h_a, w_a = mask_a.shape[:2]
    h_b, w_b = mask_b.shape[:2]

    centroid_a = get_centroid(mask_a)
    centroid_b = get_centroid(mask_b)

    if not centroid_a or not centroid_b:
        return -5.0

    cy_a, cx_a = centroid_a
    cy_b, cx_b = centroid_b
    padding = 50
    overlap_distance_y = 75
    overlap_distance_x = 60

    if side_a in ["right", "left"]:
        canvas_w = w_a + w_b + padding * 2
        canvas_h = max(h_a, h_b) + padding * 2

        offset_a_y = padding + (canvas_h - padding * 2) // 2 - cy_a
        if side_a == "right":
            offset_a_x = padding
            offset_b_x = padding + w_a - overlap_distance_x
        else:
            offset_a_x = padding + w_b - overlap_distance_x
            offset_b_x = padding

        offset_b_y = offset_a_y + cy_a - cy_b
    else:
        canvas_w = max(w_a, w_b) + padding * 2
        canvas_h = h_a + h_b + padding * 2

        offset_a_x = padding + (canvas_w - padding * 2) // 2 - cx_a

        if side_a == "bottom":
            offset_a_y = padding
            offset_b_y = padding + h_a - overlap_distance_y
        else:
            offset_a_y = padding + h_b - overlap_distance_y
            offset_b_y = padding

        offset_b_x = offset_a_x + cx_a - cx_b

    padded_a = pad_center_to_size(mask_a,
                                  (canvas_h, canvas_w),
                                  (offset_a_y, offset_a_x),
                                  )
    padded_b = pad_center_to_size(mask_b,
                                  (canvas_h, canvas_w),
                                  (offset_b_y, offset_b_x))

    combined = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    combined[padded_a > 0] = (0, 0, 255)
    combined[padded_b > 0] = (255, 0, 0)

    overlap = (padded_a > 0) & (padded_b > 0)
    combined[overlap] = (0, 255, 255)

    centroid_a_canvas = (offset_a_x + cx_a, offset_a_y + cy_a)
    centroid_b_canvas = (offset_b_x + cx_b, offset_b_y + cy_b)
    cv2.circle(combined, centroid_a_canvas, 5, (0, 255, 0), -1)
    cv2.circle(combined, centroid_b_canvas, 5, (0, 255, 0), -1)

    cv2.imshow(f"pieces aligned on {side_a} axis", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    overlap_area = np.sum(overlap)
    print(f"overlap: {overlap_area} pixels")
    return (0.75 * similarity_score) - (0.25 * overlap_area)


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
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
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
