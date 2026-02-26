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
    parser.add_argument("--record_path",
                        type=str,
                        help="path to record .mp4 video to")
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

    # init puzzler with vec environments + normalized rewards
    # that it was trained on
    puzzler = gym.make(
        "puzzler-v0",
        images=images[:5],
        seg_model_path=model_path,
        max_steps=100,
        device=DEVICE,
        render_mode="rgb_array"
    )
    puzzler = Monitor(puzzler)
    puzzler = DummyVecEnv([lambda: puzzler])
    puzzler = VecNormalize.load(
        str(project_path / "rl_env" / "train" / "vec_normalize.pkl"),
        puzzler
    )
    puzzler.training = False
    puzzler.norm_reward = False

    print(puzzler.observation_space)
    print(puzzler.action_space)

    # init env setup
    obs = puzzler.reset()
    valid_pieces = obs["valid_pieces"]
    available_pids = np.where(valid_pieces == 1)[0]
    x, y = 0, 0
    grid_size_h = puzzler.get_attr("grid_h")[0]
    i = 0

    height, width, _ = puzzler.render().shape

    # init recording, only go trough one puzzle
    if args.record_path:
        video_path = Path(args.record_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"H264"), 2, (width, height)
        )

    # if no model:
    # take from all available pieces so we don't accidently truncate
    # because of misplaced pieces
    # if model just use model information, actions are masked :P
    # PIECES ARE PLACES Y_0, Y_1, Y_2 FIRST THEN MOVE X
    if args.use_model:
        model = MaskablePPO.load(
            str(project_path / "rl_env" / "train" / "ppo_puzzler"),
            env=puzzler,
            device=DEVICE
        )
        while len(available_pids) > 0:
            print("\n\n-----------")
            action_masks = get_valid_masks(puzzler.venv.envs[0])
            action, _states = model.predict(obs,
                                            action_masks=action_masks,
                                            deterministic=True)
            obs, rewards, dones, infos = puzzler.step(action)
            print(f"reward at step {i}: {rewards[0]}")
            new_info = {
                k: v for k, v in infos[0].items()
                if k != "assembly"
            }
            print(f"INFO: {new_info}")
            frame = infos[0]["assembly"]

            if args.record_path:
                out.write(frame)

            if infos[0]["TimeLimit.truncated"]:
                print(f"\n\nPuzzler reset at step {i}.\n\n")
                available_pids = np.where(obs["valid_pieces"] == 1)[0]
                i += 1
                x, y = 0, 0
                if args.record_path:
                    break
                continue
            if dones[0]:
                print(f"\n\nPuzzler finished at step {i}.\n\n")
                cv2.imshow("final placed img", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                i += 1
                x, y = 0, 0
                if args.record_path:
                    break
                continue

            available_pids = np.where(obs["valid_pieces"] == 1)[0]

            cv2.imshow("new placed img", frame)
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
            obs, reward, dones, infos = puzzler.step(action)
            print(f"reward at step {i}: {reward}")
            new_info = {
                k: v for k, v in infos[0].items()
                if k != "assembly"
            }
            print(f"INFO: {new_info}")

            frame = infos[0]["assembly"]

            if args.record_path:
                out.write(frame)

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
            cv2.imshow("new placed img", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            y += 1
            if y >= grid_size_h:
                y = 0
                x += 1
            i += 1

    if args.record_path:
        out.release()
    puzzler.close()


if __name__ == "__main__":
    main()
