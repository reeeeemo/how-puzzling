# How Puzzling!

This is my capstone project, a culmination of everything I have learnt during my MSc at the University of Colorado Boulder.

While numerous works address the reconstruction of images from geometrically straight fragments (e.g., square, rectangular), few cover interlocking jigsaw puzzles with irregular, convex boundaries. Existing approaches for such puzzles typically depend on heuristics, manual pre-annotation, or shape-only matching without integrating modern zero-shot segmentation.

"How Puzzling!" demonstrates the feasibility of the automated reconstruction of jigsaw puzzles from images of puzzle pieces on a uniform background. This eliminates the need for human annotation by utilizing model-based segmentation with SAM 3 and YOLOv11, custom edge similarity metrics using DINOv3, and reinforcement-learning guided assembly.


## Experiments

I have listed all experiments in `.md` files across each of the folders of this project.

- [Dataset Creation via Model-Based Segmentation](./dataset/DATASET_CREATION.md)
- [Custom Edge Similarity Metrics](./similarity/SIMILARITY_CALCULATION.md)
- [Reinforcement-Learning Guided Assembly](./rl_env/RL_CREATION.md) -- **WIP as I am still training models and tuning the reward function**

## Setup

Any `.py` file that is not under `utils/` is able to be run, given the terminal instructions at the start of every file.

```bash
# if using venv
python -m venv venv
venv\Scripts\Activate

# if using anaconda
conda create --name puzzling --python=3.13
conda activate puzzling

pip install -r requirements.txt # change pytorch ver for CUDA compatibility
```