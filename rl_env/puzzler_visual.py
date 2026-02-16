import argparse
from pathlib import Path
import gymnasium as gym
import torch
import cv2
import numpy as np
from dataset.dataset import PuzzleDataset
from sb3_contrib import MaskablePPO
import random
from rl_env.train import get_valid_masks

import rl_env.env_puzzler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization of the custom gymnasium environment "puzzler"

# Use (root): python -m rl_env.puzzler_visual
#   --dataset <dataset_path>
#   --model <model_path>
#   --split <split>
#   --model <optional arg> (using a pretrained model)
# Example:
# python -m rl_env.puzzler_visual
#   --dataset dataset/data/jigsaw_puzzle
#   --model model/puzzle-segment-model/best.pt
#   --split test
#   --model


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
    parser.add_argument("--use-model",
                        action="store_true",
                        help="use trained RL model")
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
        images=images[:5],
        seg_model_path=model_path,
        max_steps=100,
        device=DEVICE
    )

    print(puzzler.observation_space)
    print(puzzler.action_space)

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
    if args.use_model:
        model = MaskablePPO.load(
            str(project_path / "rl_env" / "train" / "ppo_puzzler"),
            env=puzzler,
            device=DEVICE
        )
        while len(available_pids) > 0:
            print("\n\n-----------")
            action_masks = get_valid_masks(puzzler)
            action, _states = model.predict(obs,
                                            action_masks=action_masks,
                                            deterministic=True)
            obs, reward, term, trunc, info = puzzler.step(action)
            print(f"reward at step {i}: {reward}")
            new_obs = {k: v for k, v in obs.items() if k != "partial_assembly"}
            # print(f"OBS: {new_obs}\n\n\nINFO: {info}")
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
        while len(available_pids) > 0:
            print("\n\n-----------")
            new_pid = random.choice(available_pids)
            action = puzzler.unwrapped.coords_to_action(pid=new_pid, x=x, y=y)
            obs, reward, term, trunc, info = puzzler.step(action)
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


if __name__ == "__main__":
    main()
