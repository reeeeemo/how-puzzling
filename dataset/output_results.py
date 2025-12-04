import matplotlib.pyplot as plt
import cv2
from glob import glob
from pathlib import Path
import re

def main():
    images = sorted(glob(str(Path("") / "data" / "example_images" / "**" / "*.jpg"), recursive=True))

    # get all categories of images
    categories = {}
    prev_num = "-1"
    for img in images:
        num = Path(img).stem.split('_')
        index = '_'.join(num[2:])
        categories.setdefault(index, []).append(img)
        
    for cat in categories:
        fig, axes = plt.subplots(2,2, figsize=(24,24))
        axes = axes.ravel()
        for img_path, ax in zip(categories[cat], axes):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        fig.suptitle(f"{cat}", fontsize=32)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.03, hspace=0.03)
        plt.tight_layout()
        plt.savefig(f"./results_images/{cat}.jpg", dpi=200)
        plt.close(fig)

if __name__ == "__main__": 
    main()