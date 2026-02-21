import gymnasium as gym
import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
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


def get_valid_masks(env):
    env = env.unwrapped

    total_actions = env.num_pieces * env.grid_w * env.grid_h
    valid_mask = np.ones(total_actions, dtype=bool)

    for pid in env.current_assembled.keys():
        start_idx = pid * (env.grid_w * env.grid_h)
        end_idx = start_idx + (env.grid_w * env.grid_h)
        valid_mask[start_idx:end_idx] = False

    for y in range(env.grid_h):
        for x in range(env.grid_w):
            if env.grid[y, x] != -1:
                for pid in range(env.num_pieces):
                    action_idx = env.coords_to_action(pid=pid, x=x, y=y)
                    valid_mask[action_idx] = False
    return valid_mask


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
        images=images[:5],
        seg_model_path=model_path,
        max_steps=100,
        device=DEVICE
    )

    # THIS WILL GIVE WARNINGS ABOUT UNCONVENTIONAL SHAPE.
    # due to the edge similarity matrix + 2D grid we give PPO.
    check_env(puzzler)

    output_folder = Path(__file__).resolve().parent / "train"
    model_file = output_folder / "ppo_puzzler"
    output_folder.mkdir(parents=True, exist_ok=True)

    puzzler = ActionMasker(puzzler, get_valid_masks)
    puzzler = Monitor(puzzler)
    puzzler = DummyVecEnv([lambda: puzzler])
    puzzler = VecNormalize(puzzler, norm_obs=False)
    # create model and traiN

    if (Path(str(model_file) + ".zip")).exists():
        print("LOADING FROM MODEL FILE")
        model = MaskablePPO.load(
            str(model_file),
            env=puzzler,
            device=DEVICE
        )
    else:
        print("NEW MODEL!!!")
        model = MaskablePPO(
            "MultiInputPolicy",
            puzzler,
            learning_rate=3e-4,
            n_steps=2048,  # 2048
            n_epochs=10,
            clip_range_vf=None,
            ent_coef=0.02,  # 0.02 originally
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 256])),
            verbose=1,
            device=DEVICE,
            tensorboard_log=str(output_folder / "tb_logs" / "MaskablePPO")
        )
    model.learn(total_timesteps=500000,
                progress_bar=True,
                reset_num_timesteps=False)
    model.save(str(output_folder / "ppo_puzzler"))
    puzzler.save(str(output_folder / "vec_normalize.pkl"))


if __name__ == "__main__":
    main()
