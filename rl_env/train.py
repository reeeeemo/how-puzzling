import gymnasium as gym
import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from dataset.dataset import PuzzleDataset
import rl_env.env_puzzler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Trains a Proximal Policy Optimization (PPO) policy on the custom
# gymnasium environment "puzzler"

# Use (root): python -m rl_env.train
#   --dataset <dataset_path>
#   --model <model_path>
#   --split <split>
# Example:
# python -m rl_env.train
#   --dataset dataset/data/jigsaw_puzzle
#   --model model/puzzle-segment-model/best.pt
#   --split test


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
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parent.parent
    model_path = project_path / args.model
    dataset_path = project_path / args.dataset

    dataset = PuzzleDataset(
        root_dir=dataset_path,
        splits=[args.split],
        extension="jpg"
    )
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

    # THIS WILL GIVE WARNINGS ABOUT UNCONVENTIONAL SHAPE.
    # due to the edge similarity matrix + 2D grid we give PPO.
    check_env(puzzler)

    output_folder = Path(__file__).resolve().parent / "train"
    output_folder.mkdir(parents=True, exist_ok=True)
    # create model and traiN
    model = PPO("MultiInputPolicy",
                puzzler,
                verbose=1,
                device=DEVICE)
    model.learn(total_timesteps=10000, progress_bar=True)
    model.save(str(output_folder / "ppo_puzzler"))


if __name__ == "__main__":
    main()
