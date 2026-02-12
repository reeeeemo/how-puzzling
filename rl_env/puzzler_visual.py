import argparse
from pathlib import Path
import gymnasium as gym
import torch
import cv2
import numpy as np
from dataset.dataset import PuzzleDataset
import random

import rl_env.env_puzzler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization of the custom gymnasium environment "puzzler"

# Use (root): python -m rl_env.puzzler_visual
#   --dataset <dataset_path>
#   --model <model_path>
#   --split <split>
#   --random <true/false>
# Example:
# python -m rl_env.puzzler_visual
#   --dataset dataset/data/jigsaw_puzzle
#   --model model/puzzle-segment-model/best.pt
#   --split test
#   --random true


def main():
    parser = argparse.ArgumentParser(
        prog="Visualization of Custom Gymnasium Environment"
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
    parser.add_argument("--random",
                        type=bool,
                        required=True,
                        help="whether to place random or fixed pieces")
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parent.parent
    model_path = project_path / args.model
    dataset_path = project_path / args.dataset

    dataset = PuzzleDataset(root_dir=dataset_path,
                            splits=[args.split],
                            extension="jpg")

    images = [
        cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for img, _ in dataset
    ]
    puzzler = gym.make(
        "puzzler-v0",
        image=images[0],
        seg_model_path=model_path,
        max_steps=100,
        device=DEVICE
    )

    print(puzzler.observation_space)
    print(puzzler.action_space)

    # IMPORTANT!! If random is False, provide correct piece ID matching here.
    # Example is images/test/winter_fairy_16p.jpg if you are using the
    # provided dataset.
    random_ = args.random
    corr_pieces = [13, 14, 8, 15, 3, 12, 7, 0, 5, 11, 2, 9, 4, 6, 1, 10]

    # init env setup
    obs, info = puzzler.reset()
    valid_pieces = obs["valid_pieces"]
    available_pids = np.where(valid_pieces == 1)[0]
    x, y = 0, 0
    grid_size_h, _ = info["grid_size"]
    i = 0

    # take from all available pieces so we don't accidently truncate
    # because of misplaced pieces
    # FOR THIS EXAMPLE PIECES ARE PLACES Y_0, Y_1, Y_2 FIRST THEN MOVE X
    if random_:
        while len(available_pids) > 0:
            print("\n\n-----------")
            new_pid = random.choice(available_pids)
            obs, reward, term, trunc, info = puzzler.step((new_pid, x, y))
            print(f"reward at step {i}: {reward}")
            new_obs = {k: v for k, v in obs.items() if k != "partial_assembly"}
            print(f"OBS: {new_obs}\n\n\nINFO: {info}")
            if trunc:
                print(f"\n\nPuzzler reset at step {i}.\n\n")
                obs, info = puzzler.reset()
                available_pids = np.where(obs["valid_pieces"] == 1)[0]
                i += 1
                x, y = 0, 0
                continue
            if term:
                print(f"\n\nPuzzler finished at step {i}.\n\n")
                cv2.imshow("final placed img", obs["partial_assembly"])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                obs, info = puzzler.reset()
                i += 1
                x, y = 0, 0
                continue

            available_pids = np.where(obs["valid_pieces"] == 1)[0]
            cv2.imshow("new placed img", obs["partial_assembly"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            y += 1
            if y >= grid_size_h:
                y = 0
                x += 1
            i += 1
    else:
        for piece in corr_pieces:
            print("\n----------")
            obs, reward, term, trunc, info = puzzler.step((piece, x, y))
            print(f"reward at step {i}: {reward}")
            new_obs = {k: v for k, v in obs.items() if k != "partial_assembly"}
            print(f"OBS: {new_obs}\n\n\nINFO: {info}")
            if trunc:
                print(f"\n\nPuzzler reset at step {i}.\n\n")
                obs, info = puzzler.reset()
                i += 1
                x, y = 0, 0
                continue
            if term:
                print(f"\n\nPuzzler finished at step {i}.\n\n")
                cv2.imshow("final placed img", obs["partial_assembly"])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                obs, info = puzzler.reset()
                i += 1
                x, y = 0, 0
                continue

            cv2.imshow("new placed img", obs["partial_assembly"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            y += 1
            if y >= grid_size_h:
                y = 0
                x += 1
            i += 1


if __name__ == "__main__":
    main()
