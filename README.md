# How Puzzling!

This is my capstone project, a culmination of everything I have learnt during my MSc at the University of Colorado Boulder.

While numerous works address the reconstruction of images from geometrically straight fragments (e.g., square, rectangular) (list citations), few covers interlocking jigsaw puzzles with irregular, convex boundaries (list citations). Existing approaches for such puzzles typically depend on heuristics, manual pre-annotation, or shape-only matching without integrating modern zero-shot segmentation.

"How Puzzling!" demonstrates the feasibility of fully automated reconstruction of jigsaw puzzles from images of puzzle pieces on a uniform background. This eliminates the need for human annotation by utilizing model-based segmentation with SAM 3 (CITE), custom edge similarity metrics, and reinforcement-learning guided assembly.


## Experiments

I have listed all experiments in `.md` files across each of the folders of this project.

- [Dataset Creation via Model-Based Segmentation](./dataset/DATASET_CREATION.md)
- [Custom Edge Similarity Metrics](./similarity/SIMILARITY_CALCULATION.md)
- Reinforcement-Learning Guided Assembly (not done)