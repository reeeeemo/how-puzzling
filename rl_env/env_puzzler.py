import gymnasium as gym
from model.model import PuzzleImageModel
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from utils.polygons import create_binary_mask, get_polygon_sides
from rl_env.reward_visual import reward_function
import cv2

# Custom gymnasium environment that accepts an image and transforms the
# jigsaw puzzle into an MDP
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#subclassing-gymnasium-env


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
                 images: list[np.ndarray],
                 seg_model_path: str,
                 max_steps=100,
                 device: str = "cpu",
                 ):
        super().__init__()
        self.max_steps = max_steps
        self.device = device
        self.model = PuzzleImageModel(model_name=seg_model_path, device=device)
        self.resize_ratio = 0.5  # to resize images/masks to save memory

        self.images = [img.copy() for img in images]

        self._load_puzzle(self.images[0])

        self.grid_w, self.grid_h = self._get_grid_size()
        self.grid = np.full((self.grid_h, self.grid_w), -1, dtype=int)
        self.cur_steps = 0
        self.current_assembled = {}  # piece_id -> (row, col)

        # ex action: (piece_id, grid_x, grid_y)
        self.action_space = spaces.Discrete(
            self.num_pieces * self.grid_w * self.grid_h
        )

        ref_h = self.piece_images[0].shape[0]
        ref_w = self.piece_images[0].shape[1]

        # rl model sees cur assembled pieces, remaining ones to choose,
        # grid spots that have been placed,
        # selected piece, and candidates for selected piece
        self.observation_space = spaces.Dict({
            "partial_assembly": spaces.Box(
                low=0, high=255,
                shape=(max(ref_h*self.grid_h, 640),
                       max(ref_w*self.grid_w, 640), 3),
                dtype=np.uint8
            ),
            "remaining_pieces": spaces.Discrete(self.num_pieces + 1),
            "placed_grid": spaces.Box(
                low=-1, high=self.num_pieces-1,
                shape=(self.grid_w, self.grid_h),
                dtype=np.int64
            ),
            "selected_edge_candidates": spaces.Box(
                low=-1.0, high=1.0,
                shape=self.edge_similarities.shape,
                dtype=np.float16
            ),
            "valid_pieces": spaces.MultiBinary(self.num_pieces)
        })

    def _load_puzzle(self, image: np.ndarray):
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
        self.piece_classifications = {}
        for pid in range(self.num_pieces):
            poly = results[0].masks.xy[pid]
            box = results[0].boxes.xyxy[pid]
            self.piece_masks[pid] = create_binary_mask(
                poly=poly,
                box=box,
                img_shape=self.target_image.shape[:2],
                resize_ratio=self.resize_ratio
            )
            self.poly_sides[pid] = get_polygon_sides(
                poly=poly,
                bbox=box,
                model=self.model,
                resize_ratio=self.resize_ratio
            )
            self.piece_images[pid] = create_binary_mask(
                poly=poly,
                box=box,
                img_shape=self.target_image.shape[:2],
                image=self.target_image,
                resize_ratio=self.resize_ratio
            )
            piece_edges = [
                meta for meta in self.edges_metadata
                if meta["piece_id"] == pid
            ]
            if piece_edges:
                piece_type, piece_sides = self.model.classify_piece(
                    edge_metadata=piece_edges
                )
                self.piece_classifications[pid] = (piece_type, piece_sides)

    def _get_grid_size(self):
        """Get size of puzzle grid in width X height format."""
        num_left = len([idx for idx, types
                        in self.piece_classifications.items()
                       if "left" in types[0]])
        num_top = len([idx for idx, types
                       in self.piece_classifications.items()
                      if "top" in types[0]])
        return num_top, num_left

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
        """Computes reward of piece.

        Connectivity of pieces + positional alignment + follows puzzle rules.
        Args:
            cur_pid: current PID that was placed
        Returns:
            reward: float representing reward achieved by agent.
        """
        cardinal_directions = {
            "left": (-1, 0),
            "right": (1, 0),
            "top": (0, -1),
            "bottom": (0, 1)
        }
        opposites = {
            "top": "bottom",
            "bottom": "top",
            "right": "left",
            "left": "right"
        }
        pos_reward = 30
        cur_x, cur_y = self.current_assembled[cur_pid]
        cur_type, cur_sides = self.piece_classifications[cur_pid]
        reward = 0.0
        num_neighbors = 0

        # add positional rewards
        if cur_type.startswith("corner_"):
            flat_edges = cur_type.split("_")[1:]
            is_correct_corner = True
            for edge in flat_edges:
                if edge == "left" and cur_x != 0:
                    is_correct_corner = False
                elif edge == "right" and cur_x != self.grid_w - 1:
                    is_correct_corner = False
                elif edge == "top" and cur_y != 0:
                    is_correct_corner = False
                elif edge == "bottom" and cur_y != self.grid_h - 1:
                    is_correct_corner = False
            reward += pos_reward if is_correct_corner else -pos_reward
        elif cur_type.startswith("side_"):
            flat_edge = cur_type.split("_")[1]
            is_correct_edge = (
                (flat_edge == "left" and cur_x == 0) or
                (flat_edge == "right" and cur_x == self.grid_w - 1) or
                (flat_edge == "top" and cur_y == 0) or
                (flat_edge == "bottom" and cur_y == self.grid_h - 1)
            )
            reward += pos_reward if is_correct_edge else -pos_reward
        elif cur_type == "internal":
            is_internal_pos = (
                0 < cur_x < self.grid_w - 1 and
                0 < cur_y < self.grid_h - 1
            )
            reward += pos_reward if is_internal_pos else -pos_reward

        # for connecting pieces find edge similarity + connectivity reward
        for side, dir_vec in cardinal_directions.items():
            opposite_side = opposites[side]
            match_x, match_y = cur_x + dir_vec[0], cur_y + dir_vec[1]
            # out of bounds check
            if (match_x < 0 or
               match_x >= self.grid.shape[1] or
               match_y < 0 or
               match_y >= self.grid.shape[0]):
                continue

            match_pid = self.grid[match_y][match_x]
            if match_pid == -1:
                continue
            num_neighbors += 1

            # check puuzzle piece match eligibility
            match_type, match_sides = self.piece_classifications[match_pid]
            violations = self._check_piece_rule_eligibility(
                cur_type=cur_type,
                cur_sides=cur_sides,
                match_type=match_type,
                match_sides=match_sides,
                cur_side=side,
                opposite_side=opposite_side,
                opposites=opposites
            )
            if violations > 0:
                reward -= 30 * violations
                break

            pts_a = self.poly_sides.get(cur_pid)
            pts_b = self.poly_sides.get(match_pid)
            similarity = self._get_edge_similarity(
                piece_a_idx=cur_pid,
                side_a=side,
                piece_b_idx=match_pid,
                side_b=opposite_side
            )

            if side in pts_a and opposite_side in pts_b:
                edge_reward = reward_function(
                    mask_a=self.piece_masks.get(cur_pid),
                    mask_b=self.piece_masks.get(match_pid),
                    pts_a=pts_a,
                    pts_b=pts_b,
                    side_a=side,
                    side_b=opposites[side],
                    model=self.model,
                    similarity_score=similarity
                )
                reward += edge_reward
        if num_neighbors == 0 and len(self.current_assembled) > 1:
            reward -= 30
        return reward

    def _check_piece_rule_eligibility(self,
                                      cur_type: str,
                                      cur_sides: dict,
                                      match_type: str,
                                      match_sides: dict,
                                      cur_side: str,
                                      opposite_side: str,
                                      opposites: dict) -> int:
        """Checks if piece has violated any puzzle rules.

        Args:
            cur_type: type of current piece
            cur_sides: all side types of current piece
            match_type: type of piece to match
            match_sides: all side types of matching piece
            cur_side: current side being checked
            opposite_side: opposing side being checked
            opposites: dict of side: opposite_side
        Returns:
            num_fails: integer detailing # of fails
        """
        num_fails = 0
        # cannot compare flats or be same side type (knob-knob)
        if (
            (match_sides.get(opposite_side) == "flat") or
            (match_sides.get(opposite_side) == cur_sides.get(cur_side))
        ):
            num_fails += 1

        # side-to-side can only be same-border
        if (
            match_type.startswith("side_") and
            cur_type.startswith("side_") and
            match_type != cur_type
        ):
            num_fails += 1
        # no corner -> internal, only side pieces
        if (
            (
                match_type.startswith("corner_") and
                cur_type == "internal"
            ) or
            (
                cur_type.startswith("corner_") and
                match_type == "internal"
            )
        ):
            num_fails += 1

        # corner to corner only happens if 1 side opp and 1 side same
        if (
            cur_type.startswith("corner_") and
            match_type.startswith("corner_")
        ):
            match_flat_edges = match_type.split("_")[1:]
            cur_flat_edges = cur_type.split("_")[1:]

            intersections = set(match_flat_edges) & set(cur_flat_edges)
            opp_intersections = (
                set([opposites[side] for side in match_flat_edges]) &
                set(cur_flat_edges)
            )
            if len(intersections) <= 0 or len(opp_intersections) <= 0:
                num_fails += 1

        # corner - side can only happen if same flat side
        if (
            cur_type.startswith("corner_") and
            match_type.startswith("side_")
        ):
            match_flat_edge = match_type.split("_")[1]
            if match_flat_edge not in cur_type.split("_")[1:]:
                num_fails += 1
        elif (
            cur_type.startswith("side_") and
            match_type.startswith("corner_")
        ):
            cur_flat_edge = cur_type.split("_")[1]
            if cur_flat_edge not in match_type.split("_")[1:]:
                num_fails += 1

        # if side/internal, has to be opposite of flat
        if (
            cur_type.startswith("side_")
            and match_type == "internal"
        ):
            cur_flat_edge = cur_type.split("_")[1]
            if cur_side != opposites[cur_flat_edge]:
                num_fails += 1
        elif (
            cur_type == "internal"
            and match_type.startswith("side_")
        ):
            match_flat_edge = match_type.split("_")[1]
            if opposite_side != opposites[match_flat_edge]:
                num_fails += 1
        return num_fails

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (piece_id_to_place, x, y)

        Returns:
            tuple (observation, reward, terminated, truncated, info)
        """
        pid, x, y = self.action_to_coords(action)
        terminated, truncated = False, False

        # piece already placed,
        # max steps reached,
        # piece already in place at pos then return very bad reward
        if (
           self.current_assembled.get(pid) is not None or
           self.cur_steps >= self.max_steps or
           self.grid[y][x] != -1
           ):
            reward = -50
            truncated = True
            obs = self._get_obs()
            info = self._get_info()
            return obs, reward, terminated, truncated, info

        self.current_assembled[pid] = (x, y)
        self.grid[y][x] = pid
        self.remaining_pieces -= 1
        self.cur_steps += 1

        # reward func that consists of:
        # connectivity of pieces + position + align with puzzle rules
        reward = self._reward_function(pid)
        obs = self._get_obs()
        info = self._get_info()

        # progress reward / reward for finishing puzzle
        reward += (
            (self.num_pieces - self.remaining_pieces) /
            (self.num_pieces)
        )
        if self.remaining_pieces == 0:
            reward += 50
            terminated = True

        return obs, reward, terminated, truncated, info

    def _render_assembly(self) -> np.ndarray:
        """For each piece currently assembled, input onto the image in
        (x,y) format.

        Returns:
            ndarray: Image of all assembled puzzle pieces
        """
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

        partial_assembly = cv2.resize(partial_assembly, (640, 640))
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
        """Start a new episode by randomly selecing a new puzzle.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration
        Returns:
            tuple: (observation, info) for initial state
        """
        super().reset(seed=seed)

        # load random puzzle from list of images
        idx = self.np_random.integers(0, len(self.images))
        self._load_puzzle(self.images[idx])

        # init grid
        self.grid = np.full((self.grid_h, self.grid_w), -1, dtype=np.int64)
        self.cur_steps = 0
        self.current_assembled = {}
        self.remaining_pieces = self.num_pieces

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def action_to_coords(self, action):
        """Converts action integer to pid + coords (x,y)."""
        pid = action // (self.grid_w * self.grid_h)
        remainder = action % (self.grid_w * self.grid_h)
        x = remainder // self.grid_h
        y = remainder % self.grid_h
        return pid, x, y

    def coords_to_action(self, pid, x, y):
        """Converts pid + coords (x, y) into an action integer."""
        return pid * (self.grid_w * self.grid_h) + x * self.grid_h + y
