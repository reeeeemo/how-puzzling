import gymnasium as gym
from model.model import PuzzleImageModel
from gymnasium import spaces
import numpy as np

# some links to look into:
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#subclassing-gymnasium-env
# https://github.com/johnnycode8/gym_custom_env/blob/main/v0_warehouse_robot_env.py


class Puzzler(gym.Env):
    """Custom gymnasium-compatible environment for solving puzzles.

    Given an image, find all edges and find the matching sides.
    Attributes:
        TODO
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self,
                 puzzle_image,
                 seg_model_path: str,
                 max_steps=100,
                 device: str = "cpu",
                 render_mode=None
                 ):
        self.max_steps = max_steps
        self.model = PuzzleImageModel(model_name=seg_model_path, device=device)
        self.orig_image = puzzle_image

        (
            self.similarities,
            self.all_boxes,
            self.edges
        ) = self.model(puzzle_image)

        # a1: pick piece, a2: connect edge
        self.action_space = spaces.Discrete(2)

        # all valid edges are our states
        self.observation_space = spaces.Box(
            low=0,
            high=len(self.edges),
            shape=(1,),
            dtype=np.int32
        )

    def _get_obs(self):
        """TODO: Translate environment's state into an observation."""
        return None

    def _get_info(self):
        """TODO: Return information that is returned by step/reset."""
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO
        return None, None  # obs, info
