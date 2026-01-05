# Similarity Matching

## Bounding Box similarity:

put text here

## Segmentation similarity:

put text here

## Edge Similarity

My goal is to compute cosine similarity between each edge of a piece and all opposing edges. To do this, I need a consistent way to assign each point $p_i = (x,y)$ to one of the four global sides (top, bottom, left, right) in image coordinates.

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

> The first polar coordinate is the radial coordinate r, which is the distance of point P from the origin. The second polar coordinate is an angle $\phi$ that the radial vector makes with some chosen direction. [1]

Our origin in this case is the centroid $c$ of the polygon, found by calculating image moments of the polygon [2]. For each boundary point $p_i$, the radial vector and its unit direction are defined as:

$$\overrightarrow{r}_i= p_i - c,    \hat{r}_i = \frac{\overrightarrow{r}_i}{||\overrightarrow{r}_i||}$$

> To make up for the lack of points on straight edges using YOLO segmentations, I use the ordered polygon vertices to create a point per "step" of the mask if there is not a point already.

I can then assign the point to the side whose direction best aligns with $\hat{r}_i$

$$ side(p_i) = arg\max_{s\in sides}\hat{r}_i \cdot s $$

Both vectors are unit vectors since we normalized them, so our dot product is equivalent to the cosine of the angle between them (our second polar coordinate) [3].

I show an output [here](./dataset/results_images/radial_masks.png).

# References (TODO: cite these)
1. https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/Book%3A_University_Physics_I_-_Mechanics_Sound_Oscillations_and_Waves_(OpenStax)/02%3A_Vectors/2.05%3A__Coordinate_Systems_and_Components_of_a_Vector_(Part_2)
2. https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html
3. https://proofwiki.org/wiki/Cosine_Formula_for_Dot_Product

[1]: https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/Book%3A_University_Physics_I_-_Mechanics_Sound_Oscillations_and_Waves_(OpenStax)/02%3A_Vectors/2.05%3A__Coordinate_Systems_and_Components_of_a_Vector_(Part_2)
[2]: https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html
[3]: https://proofwiki.org/wiki/Cosine_Formula_for_Dot_Product