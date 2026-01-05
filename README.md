similarities I tried:

bbox similarity:
put text here

segment similarity:
put text here

edge similarity:

Our goal is to compute cosine similarity between every edge of a piece and all opposing edges. However, while we have the segments, we do not know the direction each (x, y) coordinate points towards.

To solve this, given a set of points of a polygon in image space, $pi = (x,y)$, we need to compute the global outward direction of each point. We are given the four cardinal directions as unit vectors:

$$
sides = {
    (0, 1), # bottom
    (0, -1), # top
    (-1, 0), # left
    (1, 0), # right
}
$$


We can actually solve this issue by looking at the definition of polar coordinates:

> The first polar coordinate is the radial coordinate r, which is the distance of point P from the origin. The second polar coordinate is an angle $\phi$ that the radial vector makes with some chosen direction. [1]

Our origin in this case is the centroid of the polygon, found by calculating the moments of a shape [2]. This is better than using the normal of a point, as that will result in a local angle, which I have calculated [an example output](./dataset/results_images/normal_masks.png). 

Each point P is represented by our (x, y) coordinates, therefore our radial coordinate $r = p_i - centroid$.

> To make up for the lack of points on straight edges using YOLO segmentations, I use the ordered polygon vertices to create a point per "step" of the mask if there is not a point already.

Using our radial found, we can take our directions inputted as unit vectors to compute the greatest degree between our point and the origin (our centroid). This results in [a nicer output](./dataset/results_images/radial_masks.png)

tangent = pi+1 - pi-1

[1]: https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/Book%3A_University_Physics_I_-_Mechanics_Sound_Oscillations_and_Waves_(OpenStax)/02%3A_Vectors/2.05%3A__Coordinate_Systems_and_Components_of_a_Vector_(Part_2)
[2]: https://docs.opencv.org/4.x/d0/d49/tutorial_moments.html