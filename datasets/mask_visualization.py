from datasets.dataset_toothsegmdataset import ToothSegmDataset, RandomGenerator


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_32_rgb_colors(cmap_name='hsv'):
    num_colors = 32
    samples = np.linspace(0, 1, num_colors, endpoint=False)
    cmap = cm.get_cmap(cmap_name, num_colors)
    rgba_colors = cmap(samples)
    rgb_colors = (rgba_colors[:, :3] * 255).astype(np.uint8)
    return rgb_colors


COLOR_32 = generate_32_rgb_colors()


def show_anns(image, input_mask, tooth_id=0, borders=True):
    """
    image: is rgb image, for visualization [h,w,3]
    input_mask: is to visulaize mask , is from testing result.  [h,w]
    tooth_id: the target tooth id for this mask.
    """
    assert tooth_id >= 0 and tooth_id < 32, "check tooth id ,it should in [0,31]"
    assert image.shape[:2] == input_mask.shape, "image and mask shape should same."
    alpha = 0.7
    color = np.array(COLOR_32[tooth_id], dtype=np.float32)
    img = image.copy()
    mask = input_mask > 2
    img[mask] = img[mask] * (1 - alpha) + color * alpha
    if borders and np.any(mask):
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        contour_color = (0, 0, 0, 0.8)
        contour_bgr = (contour_color[2], contour_color[1], contour_color[0], contour_color[3])
        cv2.drawContours(img, contours, -1, contour_bgr, thickness=1, lineType=cv2.LINE_AA)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.imshow(img)
    return img


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    base_dir = "../data/ToothSegmDataset/trainset_valset"  # adjust path
    train_dataset = ToothSegmDataset(base_dir, split="train", transform=RandomGenerator(output_size=[224, 224]))
    val_dataset = ToothSegmDataset(base_dir, split="val", transform=RandomGenerator(output_size=[224, 224]))

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # Test reading one sample
    sample = train_dataset[0]
    print("Image shape:", sample["image"].shape)  # (C,H,W)
    print("Label shape:", sample["label"].shape)  # (H,W)
    print("Case:", sample["case_name"])

    # Visualize
    image = sample["image"].permute(1, 2, 0).numpy()  # (H,W,C)
    label = sample["label"].numpy()  # (H,W)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image.astype(np.uint8))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Label")
    plt.imshow(label, cmap='jet', vmin=0, vmax=8)
    plt.axis('off')
    plt.show()

    # Visualize with show_anns
    tooth_id = 2  # example tooth id
    vis_img = show_anns(image.astype(np.uint8), label, tooth_id=tooth_id, borders=True)
    plt.figure(figsize=(5, 5))
    plt.title(f"Visualization for Tooth ID {tooth_id}")
    plt.imshow(vis_img.astype(np.uint8))
    plt.axis('off')
    plt.show()
