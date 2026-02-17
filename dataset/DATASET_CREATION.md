# Dataset Discoveries

While building the dataset for the puzzle pieces that need to segmented, I ran into an issue: I have bounding boxes, but no segmentations.

Utilizing [SAM 2](https://huggingface.co/facebook/sam2.1-hiera-base-plus) was my original solution, but with the release of [SAM 3](https://huggingface.co/facebook/sam3) I thought it would be interesting to compare the 2 models using 4 sample images from my custom dataset to figure out which will give me the more accurate segmentations. I have listed my results below.

I set up my environment following these guidelines:

- Tuning parameters for SAM3 were kept at `threshold=0.5; mask_threshold=0.5`
- Prompt Engineering (SAM3 introduced multimodality to their segmentation models, they recommend using nouns). I settled on the word `text="puzzle"` for this dataset
- Each image is projected onto a uniform background. All images were gathered from [Jigsaw Explorer](https://www.jigsawexplorer.com/). Each puzzle ranges from 9-20 pieces.
- Mask cleaning (cleaning noise from segmentations, ex. "paint splatter" effect, getting largest segmentation if multiple segmentations in same box) was done using the code below (or in [masks.py](../utils/masks.py)):

```python
def clean_masks(masks, img_shape):
    cleaned = []

    for mask in masks:
        mask_uint8 = (mask.cpu().numpy() * 255).astype(np.uint8)

        # clean noise 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea) # get largest contour

            clean_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(clean_mask, [largest], 0, 255, -1)
            cleaned.append(torch.from_numpy(clean_mask > 0))
    return torch.stack(cleaned) if cleaned else torch.empty((0, *img_shape), dtype=torch.bool)

```

## Hypothesis

A dataset consisting of larger puzzle pieces poses a challenge to a segmentation model due to the detailed **colorful images** inside of each piece. These 4 images I have tested performed the worst in the entire dataset when segmenting under any conditions, and should yield a more informative result about the segmentation method.

I have a couple of ASSUMPTIONS that I would like to prove with either segmentation method:

- If we remove the color from images, we will see a much better segmentation result.
- SAM 2 will outperform SAM 3 given a set of bounding boxes.
  - SAM 3 was built to find instance and semantic masks for objects matching a given concept [[1]]. This differs from SAM 2, which will segment anything within provided bounding boxes.
    - The bounding boxes I have annotated should provide enough information to form a mask of the entire puzzle piece, while the semantic will struggle with puzzle pieces that have larger objects embedded.

## Results

The tests I did were:

- SAM 3 with a single bounding box in the current image + text as input
  - [With color](./results_images/bbox_text.jpg)
- SAM 3 with all bounding boxes in the current image as input
  - [With color](./results_images/multi_bbox.jpg)
  - [Grayscale](./results_images/multi_bbox_gray.jpg)
  - [Only edges](./results_images/multi_bbox_edge.jpg)
- SAM 3 with all bounding boxes in the current image + text as input
  - [With color](./results_images/multi_bbox_text.jpg)
- SAM 3 with a single bounding box in the current image as input
  - [With color](./results_images/one_bbox.jpg)
  - [Grayscale](./results_images/multi_bbox_gray.jpg)
  - [Only edges](./results_images/multi_bbox_edge.jpg)
- SAM 2 with all bounding boxes in the current image as input
  - [With color](./results_images/sam2_color.jpg)
  - [Grayscale](./results_images/sam2.jpg)
  - [Grayscale + CLAHE](./results_images/sam2_clahe.jpg)
- SAM 3 with only text as input
  - [With color](./results_images/text.jpg)
  - [Grayscale](./results_images/text_gray.jpg)
  - [Grayscale + CLAHE](./results_images/text_clahe.jpg)

----

## Findings

### Grayscale vs. Edge Detecting

I found that grayscale **dramatically improves** the quality of segmentations inside of each mask, while using an edge detection filter actually **hurts** the quality.

Why does this happen? The edge detection filter, a laplacian filter, is used to find areas of rapid change in images via approximating a second derivative measurement on the image [[2]] [[6]]. This produces a sparse, high frequency image, while SAM models appear to be trained on dense, natural distributions. SAM's original paper specifically states "SAM was not trained to predict edge maps" [[3]].

This creates uncertainty amongst the structural information that the model already understands. In essence, I am asking the model to segment embeddings in a format it has never seen before in training.

----

### Why Grayscale?

Back to the grayscale, I mentioned previously that detailed and colorful images could pose a problem for the segmentation of puzzle pieces since I only want to focus on the geometry instead of the color inside the image. From my findings on the images posted above, when a puzzle piece has internal edges or colors with high variance (e.g., a puzzle piece has a small photo of a wine bottle inside, bright red apple inside of a otherwise brown puzzle), this amplifies the noise within an embedding.

I found that grayscale dulls the noise caused by color dramatically, which improved each segmentation **for my specific use case**. However if you view the grayscaled detections shown above, it still focuses on the internal edges of certain puzzle pieces (e.g., slices of bread, wine bottles, apples, cups).

----

### Improvements to Grayscale

The most interesting observation so far is that with some preprocessing, SAM 3 with a prompt like `"puzzle"` could be used as a way to gather polygons for my segmentation dataset without needing bounding boxes. Notice that the SAM 2 experiment with grayscale and a technique called CLAHE and the text-based SAM 3 experiment under the same parameters performed equally the best. There are still some small fragments missing from the segmentation, but not enough to cause distress with the model we will be training the dataset on.

Contrast Limited Adaptive Histogram Equalization, better known as CLAHE, improves the contrast of the image by dividing the image into smaller parts and performs histogram equalization independently within each tile while avoiding over-amplification of noise[[4]] [[5]]. This seems to sharpen the local features inside of each "bin", which includes the edges of every puzzle piece, but not as sharply as edge detection.

----

### Final Remarks

Overall, this seems to result in the boost of performance from CLAHE + Grayscale to maintain and enhance texture + shape within every image. The most interesting observation is that the SAM 3 experiment with a text prompt performed as well as the SAM 2 experiment with bounding boxes. This removes the limitation of each puzzle piece requiring human annotation, which may extend the length of the project. Instead, I can focus on creating a larger and more diverse dataset by inputting every image into SAM 3 with the prompt `"puzzle"`.

With all of this information, I have learned a few lessons from this experiment:

> These findings may be subjective, since it is specific to the jigsaw puzzle segmentation domain and may not generalize

- SAM 2 outperforms SAM 3 if both are only given a set of bounding boxes.
- SAM 2 with bounding boxes and SAM 3 with text input perform at a similar accuracy.
- Preprocessing both with CLAHE + Grayscale boosts accuracy dramatically, enabling a more precise segmentation of all the puzzle pieces in an image.
- Preprocessing using edge detection or another filter which creates a disruption in the natural distribution of an image lowers the accuracy given by the segmentation models.
- Puzzle pieces which contain larger artifacts inside of them (sharper internal edges, higher color variance) creates difficulty for the segmentation model, resulting in a less accurate segmentation.


With the dataset no longer requiring human annotation, I was able to focus on diversifying my dataset, and trained a YOLOv11 segmentation model which successfully segmented unseen puzzle pieces. You can find the dataset results at <https://huggingface.co/datasets/reeeemo/jigsaw_puzzle> and the model results at <https://huggingface.co/reeeemo/puzzle-segment-model>.

> Each image in the dataset was preprocessed with grayscale + CLAHE, resized to 1920x1080, and outputted as the original color image. Training was done in [train.py](../model/train.py) and inference can be done in [inference.py](../model/inference.py).

## References

1. Carion, Nicolas, et al. "Sam 3: Segment anything with concepts." arXiv preprint arXiv:2511.16719 (2025).
2. Laplacian of Gaussian Filter. (n.d.). Academic.mu.edu. <https://academic.mu.edu/phys/matthysd/web226/Lab02.htm>
3. Kirillov, Alexander, et al. "Segment anything." Proceedings of the IEEE/CVF international conference on computer vision. 2023.
4. GeeksforGeeks. (2020, May 8). CLAHE Histogram Equalization OpenCV. GeeksforGeeks. <https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/>
5. Contrast Limited Adaptive Histogram Equalization (CLAHE). (2025). @Emergentmind. <https://www.emergentmind.com/topics/contrast-limited-adaptive-histogram-equalization-clahe>
6. Spatial Filters - Laplacian/Laplacian of Gaussian. (n.d.). Homepages.inf.ed.ac.uk. <https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm>
‌

‌
‌

[1]: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
[2]: https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
[3]: https://arxiv.org/pdf/2304.02643 (7.2 / figure 10)
[4]: https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/
[5]: https://www.emergentmind.com/topics/contrast-limited-adaptive-histogram-equalization-clahe
[6]: https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm