# Dataset Discoveries
While building the dataset for the puzzle pieces that need to segmented, I ran into an issue: I have bounding boxes, but no segmentations.

Utilizing [SAM 2](https://huggingface.co/facebook/sam2.1-hiera-base-plus) was my original solution, but with the recent release of [SAM 3](https://huggingface.co/facebook/sam3) I thought it would be interesting to compare the 2 models using 4 sample images from my custom dataset to figure out which will give me the more accurate segmentations. I have listed my results below. 


Note that I have excluded a couple of conditions, most notably: 
- Tuning parameters (SAM3). These were kept at `threshold=0.5; mask_threshold=0.5`
- Prompt Engineering (SAM3 introduced multimodality to their segmentation models, they recommend using nouns). I settled on the word `text="puzzle"` for this dataset
- Mask cleaning (cleaning noise from segmentations, ex. "paint splatter" effect, getting largest segmentation if multiple segmentations in same box). The code I used for this will be below:
- Each image is projected onto a single-color background. All images were gathered from [Jigsaw Explorer](https://www.jigsawexplorer.com/). Each puzzle ranges from 15-18 pieces.

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

# Why this sample size & Hypothesis
A dataset consisting of larger puzzle pieces could seem difficult to a segmentation model due to the detailed **colorful images** inside of each piece. These 4 images I have tested performed the worst in the entire dataset when segmenting under any conditions, and should yield a more informative result about the segmentation method.

I have a couple of hypothesis' with the segmentation methods that will be put to the test:
- If we remove the color from images, we will see a much better segmentation result. 
- SAM 2 will outperform SAM 3 given a set of bounding boxes.
	+ SAM 3 was built to find semantic masks [[1]]. This differs from SAM 2, which will segment ANYTHING within the bounding box(es) provided (which will be our puzzle pieces). 
	+ As said earlier, with this method we could see puzzle pieces with larger objects inside of them perform worse than more finely detailed pieces. 

# Results

The tests I did were:
- SAM 3 with a single bounding box in the current image + text as input
    + [With color](./results_images/bbox_text.jpg)
- SAM 3 with all bounding boxes in the current image as input
    + [With color](./results_images/multi_bbox.jpg)
    + [Grayscale](./results_images/multi_bbox_gray.jpg)
    + [Only edges](./results_images/multi_bbox_edge.jpg)
- SAM 3 with all bounding boxes in the current image + text as input
    + [With color](./results_images/multi_bbox_text.jpg)
- SAM 3 with a single bounding box in the curent image as input
    + [With color](./results_images/one_bbox.jpg)
    + [Grayscale](./results_images/multi_bbox_gray.jpg)
    + [Only edges](./results_images/multi_bbox_edge.jpg)
+ SAM 2 with all bounding boxes in the current image as input
    + [With color](./results_images/sam2_color.jpg)
    + [Grayscale](./results_images/sam2.jpg)
    + [Grayscale + CLAHE](./results_images/sam2_clahe.jpg)
+ SAM 3 with only text as input
    + [With color](./results_images/text.jpg)
    + [Grayscale](./results_images/text_gray.jpg)
    + [Grayscale + CLAHE](./results_images/text_clahe.jpg)

----
**Grayscale vs. Edge Detecting**

I found that grayscale **dramatically improves** the quality of segmentations inside of each mask, while using an edge detection filter actually **hurts** the quality.

Why does this happen? My understanding is that the edge detection filter creates a binary/sparse map (low variance of active pixels), which hurts the quality since SAM models are trained on dense, natural RGB/Grayscale distributions. SAM's original paper specifically states "SAM was not trained to predict edge maps" [[2]]. This not only creates uncertainty amongst the structural information that the model already understands, but I am essentially asking the model to segment embeddings that it has never seen before in training.


----

**Why Grayscale?**

Back to the grayscale, I mentioned previously that detailed and colorful images might harm my segmentation of puzzle pieces since I only want to focus on the geometry instead of the color inside the image for my segmentation purposes. From my findings, when a puzzle piece has internal edges or colors with high variance (e.g., a puzzle piece has a small photo of a wine bottle inside, bright red apple inside of a otherwise brown puzzle), this amplifies the noise within an embedding.



I found that grayscale dulls the noise caused by color dramatically, which improved each segmentation **for my specific use case**. However if you view the grayscaled detections shown above, it still gets stuck on the internal edges of certain puzzle pieces.

----

**Improvements to Grayscale**

TODO


# References
[1]: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
[2]: https://arxiv.org/pdf/2304.02643 (7.2 / figure 10)
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5148121/
[4]: https://www.geeksforgeeks.org/python/clahe-histogram-eqalization-opencv/