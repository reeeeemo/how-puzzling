import gymnasium as gym
from model.model import PuzzleImageModel
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from utils.polygons import create_binary_mask, get_polygon_sides
from rl_env.reward_visual import reward_function

# some links to look into:
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#subclassing-gymnasium-env
# https://github.com/johnnycode8/gym_custom_env/blob/main/v0_warehouse_robot_env.py


# register as gym environment for gym.make()
register(
    id="puzzler-v0",
    entry_point="rl_env.env_puzzler:Puzzler"
)


class Puzzler(gym.Env):
    """Custom gymnasium-compatible environment for solving puzzles.

    Given an image, find all edges and find the matching sides.
    Attributes:
        TODO
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self,
                 image: np.ndarray,
                 seg_model_path: str,
                 max_steps=100,
                 device: str = "cpu",
                 ):
        super().__init__()
        self.max_steps = max_steps
        self.device = device
        self.model = PuzzleImageModel(model_name=seg_model_path, device=device)

        self.target_image = image.copy()
        results, similarities, edges_metadata = self.model([self.target_image])

        # [edge_N, edge_N, similarity]
        if hasattr(similarities, 'cpu'):
            self.edge_similarities = similarities.detach().cpu().numpy()
        else:
            self.edge_similarities = similarities

        # create mapping from (piece_idx, side) -> edge_idx
        self.edges_metadata = edges_metadata  # list of dicts of all edges info
        self.edge_idx_map = {}
        for edge_idx, edge_meta in enumerate(self.edges_metadata):
            piece_id = edge_meta["piece_id"]
            side = edge_meta["side"]
            self.edge_idx_map[(piece_id, side)] = edge_idx

        self.num_pieces = len(results[0].boxes.xyxy)
        self.remaining_pieces = self.num_pieces
        self.piece_masks = {}  # piece_id: binary mask
        self.poly_sides = {}  # piece_id: dict of side: pts
        self.piece_images = {}  # piece_id: rgb mask
        for pid in range(self.num_pieces):
            poly = results[0].masks.xy[pid]
            box = results[0].boxes.xyxy[pid]
            self.piece_masks[pid] = create_binary_mask(
                poly=poly,
                box=box,
                img_shape=self.target_image.shape[:2]
            )
            self.poly_sides[pid] = get_polygon_sides(
                poly=poly,
                bbox=box,
                model=self.model
            )
            self.piece_images[pid] = create_binary_mask(
                poly=poly,
                box=box,
                img_shape=self.target_image.shape[:2],
                image=self.target_image
            )

        self.current_assembled = {}  # piece_id -> (row, col)
        grid_size = int(np.ceil(np.sqrt(self.num_pieces)))
        self.grid = np.full((grid_size, grid_size), -1, dtype=int)
        self.cur_steps = 0

        # ex action: (piece_id, grid_x, grid_y)
        self.action_space = spaces.MultiDiscrete(
            [self.num_pieces, grid_size, grid_size]
        )

        # rl model sees cur assembled pieces, remaining ones to choose,
        # grid spots that have been placed,
        # selected piece, and candidates for selected piece
        self.observation_space = spaces.Dict({
            "partial_assembly": spaces.Box(
                low=0, high=255,
                shape=self.target_image.shape,
                dtype=np.uint8
            ),
            "remaining_pieces": spaces.Discrete(self.num_pieces + 1),
            "placed_grid": spaces.Box(
                low=-1, high=self.num_pieces-1,
                shape=(grid_size, grid_size),
                dtype=np.int64
            ),
            "selected_edge_candidates": spaces.Box(
                low=-1.0, high=1.0,
                shape=self.edge_similarities.shape,
                dtype=np.float16
            ),
            "valid_pieces": spaces.MultiBinary(self.num_pieces)
        })

    def _get_edge_similarity(
            self,
            piece_a_idx: int,
            side_a: str,
            piece_b_idx: int,
            side_b: str
         ):
        """Get similarity score between 2 pieces."""
        if (piece_a_idx, side_a) not in self.edge_idx_map:
            return None
        if (piece_b_idx, side_b) not in self.edge_idx_map:
            return None

        edge_idx_a = self.edge_idx_map[(piece_a_idx, side_a)]
        edge_idx_b = self.edge_idx_map[(piece_b_idx, side_b)]

        return self.edge_similarities[edge_idx_a, edge_idx_b].item()

    def _reward_function(self, cur_pid: int):
        """Computes reward of piece"""
        cardinal_directions = {
            "left": (-1, 0),
            "right": (1, 0),
            "top": (0, 1),
            "bottom": (0, -1)
        }
        opposites = {
            "top": "bottom",
            "bottom": "top",
            "right": "left",
            "left": "right"
        }
        cur_x, cur_y = self.current_assembled[cur_pid]
        # base = -10 for forcing curiousity, plus in faster steps
        # amplified by 10 because sim/edge_sim score is quite high
        reward = -10 - (self.cur_steps*10)

        # add sim/edge_sim for pieces that will connect
        for side, dir_vec in cardinal_directions.items():
            opposite_side = opposites[side]
            match_x, match_y = cur_x + dir_vec[0], cur_y + dir_vec[1]
            # out of bounds check
            if (match_x < 0 or
               match_x >= self.grid.shape[0] or
               match_y < 0 or
               match_y >= self.grid.shape[1]):
                continue

            match_pid = self.grid[match_y][match_x]
            if match_pid == -1:
                continue

            pts_a = self.poly_sides.get(cur_pid)
            pts_b = self.poly_sides.get(match_pid)
            similarity = self._get_edge_similarity(
                piece_a_idx=cur_pid,
                side_a=side,
                piece_b_idx=match_pid,
                side_b=opposite_side
            )

            if side in pts_a and opposite_side in pts_b:
                reward += reward_function(
                    mask_a=self.piece_masks.get(cur_pid),
                    mask_b=self.piece_masks.get(match_pid),
                    pts_a=pts_a,
                    pts_b=pts_b,
                    side_a=side,
                    side_b=opposites[side],
                    model=self.model,
                    similarity_score=similarity
                )
        return reward

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (piece_id_to_place, x, y)

        Returns:
            tuple (observation, reward, terminated, truncated, info)
        """
        pid, x, y = action
        terminated, truncated = False, False

        # piece already placed,
        # max steps reached,
        # piece already in place at pos then return bad results
        if (
           self.current_assembled.get(pid) is not None or
           self.cur_steps >= self.max_steps or
           self.grid[y][x] != -1
           ):
            reward = -1000
            truncated = True
            obs = self._get_obs()
            info = self._get_info()
            return obs, reward, terminated, truncated, info

        self.current_assembled[pid] = (x, y)
        self.grid[y][x] = pid
        self.remaining_pieces -= 1
        self.cur_steps += 1

        # reward function is going to be all 4 cardinal directions
        # we need to get the connectivity of them all (piecemask to piecemask)
        # so edge distance + sim score
        reward = self._reward_function(pid)
        obs = self._get_obs()
        info = self._get_info()

        if self.remaining_pieces == 0:
            terminated = True
            reward += 1000
        return obs, reward, terminated, truncated, info

    def _render_assembly(self) -> np.ndarray:
        """For each piece currently assembled, input onto the image in
        (x,y) format.

        Returns:
            ndarray: Image of all assembled puzzle pieces
        """
        # partial_assembly = np.zeros(self.target_image.shape, dtype=np.uint8)
        ref_h = self.piece_images[0].shape[0]
        ref_w = self.piece_images[0].shape[1]

        grid_h, grid_w = self.grid.shape[:2]
        partial_assembly = np.zeros((grid_h*ref_h, grid_w*ref_w, 3),
                                    dtype=np.uint8)

        margin_x = ref_w // 2
        margin_y = ref_h // 2

        for pid, (x, y) in self.current_assembled.items():
            mask = self.piece_images[pid]

            centroid = self.model.get_centroid(self.piece_masks[pid],
                                               binary_mask=True)
            cx, cy = int(centroid[0]), int(centroid[1])
            h, w = mask.shape[:2]

            y_start = int(y * ref_h) - cy + margin_y
            x_start = int(x * ref_w) - cx + margin_x

            src_y1 = max(0, -y_start)
            src_x1 = max(0, -x_start)
            dst_y1 = max(0, y_start)
            dst_x1 = max(0, x_start)

            dst_y2 = min(partial_assembly.shape[0], y_start + h)
            dst_x2 = min(partial_assembly.shape[1], x_start + w)

            src_y2 = src_y1 + (dst_y2 - dst_y1)
            src_x2 = src_x1 + (dst_x2 - dst_x1)

            mask_crop = mask[src_y1:src_y2, src_x1:src_x2]
            piece_mask = (mask_crop > 0).any(axis=2)
            dest_region = partial_assembly[dst_y1:dst_y2, dst_x1:dst_x2]
            dest_region[piece_mask] = mask_crop[piece_mask]

        return partial_assembly

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        valid_pieces_mask = np.ones(self.num_pieces, dtype=np.int8)
        for pid in self.current_assembled.keys():
            valid_pieces_mask[pid] = 0

        return {
            "partial_assembly": self._render_assembly(),
            "remaining_pieces": self.remaining_pieces,
            "placed_grid": self.grid,
            "selected_edge_candidates": self.edge_similarities,
            "valid_pieces": valid_pieces_mask
        }

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        # maybe at some point add human rendering so return partial assembly?
        return {
            "num_pieces": self.num_pieces,
            "remaining_pieces": self.remaining_pieces,
            "total_eligible_edges": len(
                [meta for meta in self.edges_metadata
                 if meta["side_type"] != "flat"]
            ),
            "total_similarities": len(self.edge_similarities),
            "grid_size": self.grid.shape
        }

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration
        Returns:
            tuple: (observation, info) for initial state
        """
        super().reset(seed=seed)

        # init grid
        grid_size = int(np.ceil(np.sqrt(self.num_pieces)))
        self.grid = np.full((grid_size, grid_size), -1, dtype=int)
        self.cur_steps = 0
        self.current_assembled = {}
        self.remaining_pieces = self.num_pieces

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
