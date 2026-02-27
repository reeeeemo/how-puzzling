# RL Guided Assembly via MaskablePPO

My final leg of the automated reconstruction of jigsaw puzzles is the reinforcement-learning guided assembly.

Given an image of separated puzzle pieces, the model should be able to:

**1. Segment all puzzle pieces, approximate their sides correctly, and gather metrics**
2. Create a Markov Decision Process $<S, A, r, P, p_0>$ for the RL model
3. Train and utilize a model to successfully place the puzzle pieces at the correct coordinates as efficiently as possible.

---

To figure out the best model type and whether RL is feasible, I need to create an MDP for this problem. Before getting started however, I must define a set of rules whenever any model (heuristics, RL) attempts to solve a jigsaw puzzle:

> [SIMILARITY_CALCULATION.md](../similarity/SIMILARITY_CALCULATION.md) describes some clarifications that apply here as well, especially how each piece and edge is classified.

1. A puzzle piece edge may only fit together if the opposing edge is the opposite edge type(knob-hole, hole-knob, NEVER knob-knob or hole-hole).
2. A puzzle piece edge that is classified as "flat" is invalid to have any connecting edge.
3. Each puzzle piece that is classified as a "corner" piece may only connect to "side" or other "corner" pieces
4. Each puzzle piece that is classified as a "side" piece may only connect to "corner" and other "side" pieces that share a flat side, and can connect to "internal" pieces as long as the edge being connected is opposite the flat side.
5. Any "corner" piece's x and y position on a coordinate grid will be either a minimum or maximum point (e.g., top-left would be <0, 0>).
6. Any "side" piece's position must contain at least 1 minimum or maximum point (e.g., sides with a flat top will be <1, 0>)
7. Any "internal" piece's position will not contain ANY minimum or maximum points.

With that in mind, I can now create an MDP.

## Markov Decision Process

For this entire discussion, I will be using a 3x3 jigsaw puzzle as an example, that means 9 total pieces. 

> *n* and *m* will represent the puzzle piece width/height

### S

The set of all valid states depends on the number of puzzle pieces and grid positions. For my example, this means 362,880 states ($9!$ total valid states).

This is factorial growth, since everytime the agent places a piece, the number of unique pieces and grid positions decreases by 1. Further puzzle sizes (4x4, 5x5) can grow these to even larger values. This introduces the curse of dimensionality and increases time/space complexity when using techniques like Q-Learning and DQNs.

$$ S = (n \cdot m)! $$ 

### A

The set of all valid actions also depends on the number of puzzle pieces and grid positions. For my example, this means 81 potential actions.

For further puzzle sizes (4x4, 5x5), this expands to 256, 625 actions. This is a huge amount of potential actions for the agent to explore, whose performance in this situation might be increased by masking invalid actions over time.

$$ |A| = (n \cdot m)^2 $$

### r

The reward function introduced some difficulty. The initial idea for the function was:

$$ r = (-10 \cdot (steps_{current} \cdot 10)) + ((edge_{similarity} \cdot \mu) + (edge_{overlap} \cdot \mu)) $$

Essentially, the reward was calculated by placing a piece, checking every cardinal direction for a valid piece, and checking it's similarity and overlap when the pieces were placed together. $\mu$ Is a multiple that determines the weight of the edge-to-edge similarity between every piece.

While this was effective in determining an initial model's attempt at solving a puzzle, after extensive research, a few issues with the reward function arose:

- Giving negative rewards per-step to a maskable action model like MaskablePPO proved to be ineffective, since there was only a limited amount of options to take anyways.
    - Additionally, a small progressive reward seemed to improve the model results.
- [similarity.py](../similarity/similarity.py) shows how to get rid of puzzle pieces that cannot fit together under the rules described earlier. These are not addressed in the reward function
    - To add on, an isolation penalty was implemented such that it would not default to incorrect pieces being placed in spots with no other pieces, to disallow for the agent to "cheat" a positive reward
- After implementing these fixes, I noticed puzzles can have high variance due to the nature of the puzzle rules rewards and similarity rewards. While extensive tuning was done to ensure rewards were not overbearing, normalization seemed to stabilize the model.

The reward function, found in [env_puzzler.py](./env_puzzler.py), is mostly computed in `reward_function()` while the violations multiple $\phi$ is computed in `_check_piece_rule_eligibility()`:

$$ r = (R_{progress} + R_{positional}) + R_{completed} - (10 \cdot \phi) - R_{isolated} + ((edge_{similarity} \cdot \mu) + (edge_{overlap} \cdot \mu))\cdot edges_{connected} $$

- $R_{positional}$ is -10 for failures, and +30 for successful positions
- $R_{progress}$ is calculated by taking the number of current pieces remaining by total
- $R_{completed}$ represents the positive reward given if a puzzle has no more available pieces to place.
- $R_{isolated}$ is the small negative reward for placing pieces with no connecting edges.
- $edges_{connected}$ is the number of edges that the currently placed piece is connected to.

Finally, since PPO is a model-free algorithm, the transition function $P$ is not explicitly defined.

## Environment Creation

To solve this MDP, I created a custom gymnasium environment called "puzzler-v0".

### Observation State
The observation state includes:

- Count of remaining piece IDs
- Array of assembled puzzle pieces compressed into a one-dimensional format.
    - Using `[x, y] = piece IDs` for this gave confusing signals to the agent, so it was changed to `[x, y] = piece features`
- Compressed 2D array of a cosine similarity matrix for each edge of every puzzle piece
    - This was also changed to `[pid_a, pid_b, side_idx] = sim` so that the agent could correlate piece ID with the piece features/the piece its placing.
    - These scores are precomputed, and further explained in [SIMILARITY_CALCULATION.md](../similarity/SIMILARITY_CALCULATION.md)
- Array of piece features consisting of `[is_corner, is_side, is_internal, has_left, has_right, has_top, has_bottom]` for every piece.
- Binary mask of valid piece IDs

Originally an image of the partially reconstructed puzzle was provided, but it provided to only generate a weak signal and longer training compared to the puzzle piece features given.

### Action Space
The action space originally utilized `[pid, x, y]` as a valid action, however MaskablePPO does not support Box or Dict actions(multi-dimensional arrays) [[1]]. This lead me to create an equation for computing an action given a pid, x, and y coordinate in one array. It is found in [env_puzzler.py](./env_puzzler.py) as `action_to_coords` and `coords_to_action`:

Conversion to action:

$$ action = pid \cdot (width \cdot height) + x * height + y $$

Conversion back to `[pid, x, y]`:

$$ pid = \lfloor \frac{action}{width \cdot height} \rfloor $$

$$ x = \frac{action \mod (width \cdot height)}{height} $$

$$ y = (action \mod (width \cdot height)) \mod height $$ 

---
Something to note is that due to the nature of neural networks requiring static input/output tensor shapes, it is impossible to train on a 3x3 puzzle and run inference on a 4x4 puzzle with the same weights.

The environment also supports multiple images, that are randomly selected during the `reset()` function, it expects all puzzle images to be the same amount of pieces and grid shape (3x3, 4x4, etc.).

I also used VecNormalize [[2]] to normalize the rewards, as stated earlier, for less variance during training.


## Model Training / Outcomes

> All of the training code/hyperparameters can be found in [train.py](./train.py). Including the action masking.

The majority of time spent on this project was spent on the training of each model and tuning of the reward function. An example video, weights, and training data can be found in the [huggingface repo](https://huggingface.co/reeeemo/ppo-puzzler). 

The final model I posted on huggingface is able to solve the puzzle, however it has a small percentage of misplacing a piece still. Further considerations for a perfect reconstruction could be solved with extra training, stacking gymnasium environments and including further images (training size was 5 images). These require compute resources beyond my current capabilities.

Overall, even with the current model, I believe I have demonstrated the feasibility of the automated reconstruction of jigsaw puzzles from images of puzzle pieces, eliminating the need for human annotation. If this was to be used as real-world software, I would recommend a heuristic approach using the edge similarity metrics. This is due to the RL model's inability to generalize for every puzzle size.

## References

1. Maskable PPO — Stable Baselines3 - Contrib 2.8.0a1 documentation. (2021). Readthedocs.io. <https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html>
2. Vectorized Environments — Stable Baselines3 2.8.0a4 documentation. (2021). Readthedocs.io. <https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html>


[1]: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
[2]: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html