import argparse
from pathlib import Path
import gymnasium as gym
import torch
import cv2
import numpy as np
from dataset.dataset import PuzzleDataset
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import random
from rl_env.train import get_valid_masks
import warnings

import rl_env.env_puzzler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization of the custom gymnasium environment "puzzler"
# Note that the recording software uses Windows Media Foundation's CAP_MSMF
# instead of libopenh264 using the ffmpeg backend :p

# Use (root): python -m rl_env.puzzler_visual
#   --dataset <dataset_path>
#   --model <model_path>
#   --split <split>
#   --rl_model <rl_model_path>
#   --normalized_env <normalized_environment>
#   --record_path <replay_path>
# Example:
# python -m rl_env.puzzler_visual
#   --dataset dataset/data/jigsaw_puzzle
#   --model model/puzzle-segment-model/best.pt
#   --split test
#   --rl_model rl_env/train/ppo-puzzler/ppo_puzzler
#   --normalized_env rl_env/train/ppo-puzzler/vec_normalize.pkl
#   --record_path replay.mp4

def parse_args():
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
    parser.add_argument("--rl_model",
                        type=str,
                        help="use trained RL model")
    parser.add_argument("--normalized_env",
                        type=str,
                        help="saved normalized stats for environment")
    parser.add_argument("--record_path",
                        type=str,
                        help="path to record .mp4 video to")
    return parser.parse_args()

def start_puzzle_reconstruction(
        env: gym.Env, 
        model: MaskablePPO,
        video_writer: cv2.VideoWriter,
        available_pids: np.ndarray
    ):
    """Start puzzle reconstruction. Record and use model if possible.

    Places pieces first on the y-axis, then moves across the x-axis.

    Args:
        env: gymnasium environment to reconstruct puzzles with
        model: model to predict which piece to place if given
        video_writer: record and write video if given
        available_pids: PIDs that are able to be placed
    """
    x, y, i = 0, 0, 0
    grid_size_h = env.get_attr("grid_h")[0]
    obs = env.reset()
    try:
        # take from all available pieces if not using a model.
        # if model then MaskablePPO sorts out the available pieces issue.
        while len(available_pids) > 0:
            print("\n\n----------")
            if model:
                action_masks = get_valid_masks(env.venv.envs[0])
                action, _ = model.predict(
                    obs,
                    action_masks=action_masks,
                    deterministic=True
                )
            else:
                new_pid = random.choice(available_pids)
                action = env.unwrapped.coords_to_action(pid=new_pid, x=x, y=y)
            obs, reward, dones, infos = env.step(action)
            
            print(f"reward at step {i}: {reward}")
            new_info = {
                k: v for k, v in infos[0].items()
                if k != "assembly"
            }
            print(f"INFO: {new_info}")

            frame = infos[0]["assembly"]
            if video_writer:
                video_writer.write(frame)

            # if truncated or terminated
            if infos[0]["TimeLimit.truncated"]:
                print(f"\n\nPuzzler reset at step {i}.\n\n")
                available_pids = np.where(obs["valid_pieces"] == 1)[0]
                i += 1
                x, y = 0, 0
                continue
            if dones[0]:
                print(f"\n\nPuzzler finished at step {i}.\n\n")
                cv2.imshow("final placed img", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                i += 1
                x, y = 0, 0
                continue

            available_pids = np.where(obs["valid_pieces"] == 1)[0]

            cv2.imshow("new placed image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            y += 1
            if y >= grid_size_h:
                y = 0
                x += 1
            i += 1
    finally:
        if video_writer:
            video_writer.release()
        env.close()
    

def main():
    print(f"device: {DEVICE}")
    args = parse_args()

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

    # init puzzler with vec environments + normalized rewards
    puzzler = gym.make(
        "puzzler-v0",
        images=images[:6],
        seg_model_path=model_path,
        max_steps=100,
        device=DEVICE,
        render_mode="rgb_array"
    )
    puzzler = Monitor(puzzler)
    puzzler = DummyVecEnv([lambda: puzzler])
    if args.normalized_env:
        puzzler = VecNormalize.load(
            str(project_path / args.normalized_env),
            puzzler
        )
    else:
        warnings.warn(
            """
            No VecNormalized trained environment was given. 
            Creating new environment...
            """
        )
        puzzler = VecNormalize(puzzler)

    puzzler.training = False
    puzzler.norm_reward = False

    print(puzzler.observation_space)
    print(puzzler.action_space)

    # init env setup
    obs = puzzler.reset()
    valid_pieces = obs["valid_pieces"]
    available_pids = np.where(valid_pieces == 1)[0]

    height, width, _ = puzzler.render().shape

    # init recording
    if args.record_path:
        video_path = Path(args.record_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(video_path), cv2.CAP_MSMF,
            cv2.VideoWriter_fourcc(*"H264"), 2, (width, height)
        )
    if args.rl_model:
        model = MaskablePPO.load(
            str(project_path / args.rl_model),
            env=puzzler,
            device=DEVICE
        )

    start_puzzle_reconstruction(
        env=puzzler,
        model=model,
        video_writer=out,
        available_pids=available_pids
    )
    
if __name__ == "__main__":
    main()
