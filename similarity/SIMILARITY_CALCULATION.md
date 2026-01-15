# Similarity Piece-To-Piece

After getting successful segmentations, I shifted my focus over to pairwise cosine similarity between my jigsaw puzzle pieces using a [DINOv3 image encoder](https://huggingface.co/papers/2508.10104). However, I need cropped images for this problem, so I attempted 3 measures of similarity.

1. Bounding Box Similarity
    - Little variance, as all puzzle pieces scored above the 0.950 percentile
    - Background and relative colors across the entire puzzle could be a factor for this increase in similarity
2. Segmentation Similarity
    - Better variance (.300 difference)
    - Is not shape-agnostic, which causes similar puzzle pieces which do not fit together to be the most similar.
    - Does not solve the issue of knowing the placement of the piece relative to the selected piece. For example, a middle 4 piece versus another middle 4 piece, which side will fit it best?

This brings me to my final and working method, edge-to-edge similarity.

## Edge Similarity

My goal is to compute cosine similarity between each edge of a piece and all opposing edges. To do this, I need a consistent way to assign each point $p_i = (x,y)$ to one of the four global sides (top, bottom, left, right) in image coordinates.

> I have created a [Desmos graph](https://www.desmos.com/geometry/ea0brhxr2z) of the side approximation algorithm I created for this problem. Please take a look if anything does not make a sense!

I will represent the four cardinal directions as unit vectors. All directions are defined in image coordinates, where the y-axis increases downwards:

```python
sides = {
    (0, 1), # bottom
    (0, -1), # top
    (-1, 0), # left
    (1, 0), # right
}
```

My first idea was to use the local normal of the point(perpendicular to the tangent). [This is the result of my calculations](./dataset/results_images/normal_masks.png). As you can see, the flat edges produce the correct direction while any point in the tab/holes of a jigsaw puzzle piece assigns different directions.

Instead, I opted to solve this issue by using *polar coordinates*:

> The first polar coordinate is the radial coordinate r, which is the distance of point P from the origin. The second polar coordinate is an angle $\phi$ that the radial vector makes with some chosen direction. [[1]]

Our origin in this case is the centroid $c$ of the polygon, found by calculating image moments of the polygon [[2]] [[4]]. For each boundary point $p_i$, the radial vector and its unit direction are defined as:

$$\overrightarrow{r}_i= p_i - c,    \hat{r}_i = \frac{\overrightarrow{r}_i}{||\overrightarrow{r}_i||}$$

> To make up for the lack of points on straight edges using YOLO segmentations, I densifed the convex polygon such that every step of 1.0 has a new point for the most accurate centroid approximation.

I can then assign the point to the side whose direction best aligns with $\hat{r}_i$

$$ side(p_i) = arg\max_{s\in sides}\hat{r}_i \cdot s $$

Both vectors are unit vectors since we normalized them, so our dot product is equivalent to the cosine of the angle between them (our second polar coordinate) [[3]].

I have calculated an output [to this image output for the global process](./dataset/results_images/radial_masks.png).

## Results

This worked with rectangular crops, however...
1. Each similarity was not consistent and regularly did not place correct pieces in the top 5 ranking of each edge as I had hoped. 
2. Rectangular crops also take a part of the adjacent sides (right crop would have portions of the top/bottom crop) which poisons each edge's similarity metrics.

The solution was relatively simple, take the current side's mask created by the segmentation model and compute the inner point given an integer that defines how much of the width/height I want from the edge.

> I have created another [Desmos graph](https://www.desmos.com/geometry/zjzaztcfcv) of this inner point calculation, move around the sliders!

To approximate the local inward normal, I first compute a central-difference approximation of the tangent vector between 2 neighboring boundary points [5]:

$$T = -\frac{(p_{n+1}-p_{n-1})}{\Delta{t}}$$

To ensure the direction points inward, I take the dot product between the polygon centroid and $T$ and negate if necessary to get $p_{inward}$. I then compute our inner point given an integer edge width $e_{width}$ and the original point $p_i$

$$p_{inner}=p_i+p_{inward}\cdot(\frac{e_{width}}{2}) $$

I have also added some morphological operations to fix any artifacts in the image and a flat-edge detector to abstract any pieces that do not fit with any piece. As a result, I have observed consistent top 5 rankings for each correct edge given a puzzle piece to compare. [Here is an example](./dataset/results_images/EdgeMatches.png). 

While I do have issues with some inner points creating more artifacts (see the desmos graph and slide the points aroundfor an example), this does not harm the similarity rankings enough to cause issues. This also does not remove the shape similarities all together, however it does reduce their impact on the rankings enough to give correct results!

## References (TODO: cite these)

1. Libretexts. (2024, October 1). 2.5: Coordinate systems and components of a vector (part 2). Physics LibreTexts. <https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/Book%3A_University_Physics_I_-_Mechanics_Sound_Oscillations_and_Waves_(OpenStax)/02%3A_Vectors/2.05%3A__Coordinate_Systems_and_Components_of_a_Vector_(Part_2)>
2. Huamán, A. (n.d.). Goal. OpenCV. <https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html>
3. Cosine Formula for Dot Product - ProofWiki. (2023). Proofwiki.org. <https://proofwiki.org/wiki/Cosine_Formula_for_Dot_Product>
4. Dayala, R. (2020, July 21). 10.4 Hu Moments. Computer Vision. https://cvexplained.wordpress.com/2020/07/21/10-4-hu-moments/
5. Djellouli, A. (2024). Central Difference Method. Adamdjellouli.com. https://adamdjellouli.com/articles/numerical_methods/3_differentiation/central_difference

‌

[1]: https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/Book%3A_University_Physics_I_-_Mechanics_Sound_Oscillations_and_Waves_(OpenStax)/02%3A_Vectors/2.05%3A__Coordinate_Systems_and_Components_of_a_Vector_(Part_2)
[2]: https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html
[3]: https://proofwiki.org/wiki/Cosine_Formula_for_Dot_Product
[4]: https://cvexplained.wordpress.com/2020/07/21/10-4-hu-moments/
[5]: https://adamdjellouli.com/articles/numerical_methods/3_differentiation/central_difference
